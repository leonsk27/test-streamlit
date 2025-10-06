import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ---------- Utilidades ----------
def read_csv_smart(path: Path) -> pd.DataFrame:
    # Intenta detectar separador automáticamente
    for sep in (",", ";", "\t", None):
        try:
            if sep is None:
                return pd.read_csv(path, engine="python")
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    raise RuntimeError(f"No pude leer el CSV: {path}")

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_catalog_maps(items: list):
    prod_by_id = {}
    cat_by_id  = {}
    code_by_id = {}
    for it in items:
        pid = it.get("id")
        if pid is not None:
            prod_by_id[pid] = it.get("nombre")
            code_by_id[pid] = it.get("codigo")
        cat = it.get("categoria") or {}
        cid = cat.get("id")
        if cid is not None:
            cat_by_id[cid] = cat.get("nombre")
    return prod_by_id, cat_by_id, code_by_id

def ensure_stable_mapping(generic_ids, real_ids, mapping: dict, key: str):
    """
    Asigna IDs reales a genéricos que no estén mapeados, de forma determinista,
    y persiste en 'mapping[key]'.
    """
    mapping.setdefault(key, {})
    m = mapping[key]
    real_sorted = list(sorted(set(real_ids)))
    if not real_sorted:
        return m

    # Reutiliza los ya asignados; para nuevos, round-robin
    i = 0
    for g in sorted(set(str(x) for x in generic_ids if pd.notna(x))):
        if g not in m:
            # si el genérico coincide con un real, úsalo tal cual
            try:
                gi = int(g)
                if gi in real_sorted:
                    m[g] = gi
                    continue
            except Exception:
                pass
            m[g] = real_sorted[i % len(real_sorted)]
            i += 1
    return m

def first_col(df: pd.DataFrame, candidates: tuple):
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None

# ---------- Script principal ----------
def main():
    ap = argparse.ArgumentParser(description="Adaptar ventas_simuladas.csv con nombres reales y encabezados claros.")
    ap.add_argument("--csv-in",  required=True, type=Path, help="Ruta del CSV de ventas (simulado/original).")
    ap.add_argument("--csv-out", required=True, type=Path, help="Ruta de salida del CSV adaptado (puede ser el mismo).")
    ap.add_argument("--catalog", required=True, type=Path, help="JSON del catálogo del servidor (arreglo como el que pegaste).")
    ap.add_argument("--mapping", type=Path, default=Path("data/mapeos/mapeo_ids.json"), help="JSON para persistir mapeo estable.")
    ap.add_argument("--backup",  action="store_true", help="Crear backup del CSV de entrada antes de sobrescribir.")
    ap.add_argument("--aggregate-daily", action="store_true", help="Si hay varias filas por fecha, suma ventas por día.")
    args = ap.parse_args()

    if args.backup and args.csv_in.resolve() == args.csv_out.resolve():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = args.csv_in.with_suffix(f".{ts}.bak.csv")
        backup_path.write_bytes(args.csv_in.read_bytes())
        print(f"[i] Backup creado: {backup_path}")

    # 1) Catálogo (id->nombre)
    items = load_json(args.catalog)
    prod_by_id, cat_by_id, code_by_id = build_catalog_maps(items)
    print(f"[i] Catálogo: {len(prod_by_id)} productos, {len(cat_by_id)} categorías")

    # 2) CSV original
    df = read_csv_smart(args.csv_in)
    orig_cols = df.columns.tolist()

    # 3) Detectar columnas
    date_col  = first_col(df, ("ds","fecha","date","fecha_venta"))
    value_col = first_col(df, ("y","ventas","venta","monto","recaudacion","cantidad","valor"))
    prod_col  = first_col(df, ("producto_id","product_id","id_producto","producto"))
    cat_col   = first_col(df, ("categoria_id","id_categoria","categoria","cat_id"))

    if date_col is None:
        sys.exit("No encontré una columna de fecha (ds/fecha/date/fecha_venta)")
    if value_col is None:
        sys.exit("No encontré una columna de ventas (y/ventas/venta/monto/recaudacion/cantidad/valor)")

    # 4) Normalizar fecha/ventas + renombrar encabezados
    df = df.rename(columns={date_col: "fecha_venta", value_col: "ventas"})

    # 5) Mapear producto/categoría → nombres reales (congruencia persistente)
    mapping = {"producto": {}, "categoria": {}}
    if args.mapping.exists():
        try:
            mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        except Exception:
            pass

    if prod_col:
        # si renombramos value/date arriba, aún existe prod_col con su nombre original
        genericos_prod = df[prod_col] if prod_col in df.columns else None
        if genericos_prod is not None:
            ensure_stable_mapping(genericos_prod, list(prod_by_id.keys()), mapping, "producto")
            # Real ID para cada fila
            real_prod_id = genericos_prod.astype("string").map(mapping["producto"]).astype("Int64")
            df["producto_id_real"] = real_prod_id
            df["producto_nombre"]  = df["producto_id_real"].map(prod_by_id)
            df["codigo"]           = df["producto_id_real"].map(code_by_id)

    if cat_col:
        genericos_cat = df[cat_col] if cat_col in df.columns else None
        if genericos_cat is not None:
            ensure_stable_mapping(genericos_cat, list(cat_by_id.keys()), mapping, "categoria")
            real_cat_id = genericos_cat.astype("string").map(mapping["categoria"]).astype("Int64")
            df["categoria_id_real"] = real_cat_id
            df["categoria_nombre"]  = df["categoria_id_real"].map(cat_by_id)

    # 6) Guardar el mapeo actualizado
    save_json(args.mapping, mapping)
    print(f"[i] Mapeo persistido en: {args.mapping}")

    # 7) (Opcional) Agregar por día si se pide
    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce")
    df = df.dropna(subset=["fecha_venta"])
    if args.aggregate_daily:
        # Mantiene columnas descriptivas si son constantes por fecha; si no, se descartan en el agregado
        keep_cols = [c for c in ("producto_nombre","categoria_nombre","codigo") if c in df.columns]
        grouped = df.groupby(df["fecha_venta"].dt.normalize(), as_index=False)["ventas"].sum()
        grouped = grouped.rename(columns={"fecha_venta":"fecha_venta","ventas":"ventas"})
        df = grouped  # simple y seguro para Prophet

    # 8) Ordena columnas amigables (si existen)
    preferred = [c for c in ["fecha_venta","ventas","producto_nombre","categoria_nombre","codigo"] if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    df = df[preferred + rest]

    # 9) Guardar CSV adaptado
    write_csv(df, args.csv_out)
    print(f"[✔] CSV adaptado guardado en: {args.csv_out}")
    print(f"[i] Columnas: {df.columns.tolist()}")

if __name__ == "__main__":
    main()

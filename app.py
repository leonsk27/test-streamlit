# app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np
# Compat con NumPy 2.x para Prophet (debe ir ANTES del import de prophet)
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "complex_"): np.complex_ = np.complex128
if not hasattr(np, "int_"):     np.int_     = np.int64

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px

# ---------- Config de página ----------
st.set_page_config(page_title="Predicción de Ventas", layout="wide")
st.title("📈 Predicción de ventas de suministros")
st.markdown("Lectura automática desde `data/ventas_simuladas.csv` (o variable de entorno `VENTAS_CSV_PATH`).")

# ---------- Sidebar: opciones del modelo / limpieza ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    horizon_days = st.slider("Horizonte de pronóstico (días)", 14, 365, 90, step=7)
    weekly_season = st.checkbox("Estacionalidad semanal", value=True)
    yearly_season = st.checkbox("Estacionalidad anual", value=False)
    changepoint_scale = st.slider("Sensibilidad a cambios (changepoint_prior_scale)", 0.01, 1.0, 0.2)
    growth_type = st.selectbox("Tipo de tendencia", ["linear", "flat", "logistic"], index=0)
    agg_to_daily = st.checkbox("Agregar por día (si hay fechas repetidas)", value=True)
    fill_missing_as_zero = st.checkbox("Rellenar días faltantes con 0 ventas", value=True)

# ---------- Lectura automática del CSV ----------
DATA_PATH = Path(os.getenv("VENTAS_CSV_PATH", "data/ventas_simuladas.csv")).expanduser()
st.caption(f"📄 Fuente de datos: **{DATA_PATH.resolve()}**")

@st.cache_data(show_spinner=False)
def read_csv_smart(path: Path):
    # intenta varios separadores y engine python si hace falta
    for sep in (",", ";", "\t", None):
        try:
            if sep is None:
                return pd.read_csv(path, engine="python")
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    raise RuntimeError(f"No pude leer el CSV: {path}")

if not DATA_PATH.exists():
    st.error(f"No encuentro el archivo en: {DATA_PATH.resolve()}")
    st.stop()

# Usamos el mtime como parte del cache-key para auto-recarga si el archivo cambia
mtime = DATA_PATH.stat().st_mtime
df_raw = read_csv_smart(DATA_PATH)

st.subheader("Vista previa")
st.dataframe(df_raw.head(10), use_container_width=True)

# ---------- Detección de columnas ----------
def pick_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None

# Preferimos encabezados "bonitos" si ya usaste el preprocesador
date_col  = pick_col(df_raw, ("fecha_venta","fecha","date","ds"))
value_col = pick_col(df_raw, ("ventas","y","venta","monto","cantidad","valor"))

if date_col is None or value_col is None:
    st.error("No pude detectar columnas de fecha/ventas. Esperaba `fecha_venta`/`ventas` o `ds`/`y`.")
    st.stop()

# ---------- Limpieza a formato Prophet (ds, y) ----------
df = df_raw[[date_col, value_col]].copy()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["ds", "y"]).sort_values("ds")

if agg_to_daily:
    df["ds"] = df["ds"].dt.normalize()
    df = df.groupby("ds", as_index=False)["y"].sum()

if fill_missing_as_zero:
    full_idx = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_idx)
    df.index.name = "ds"
    df["y"] = df["y"].fillna(0.0)
    df = df.reset_index()

if len(df) < 10:
    st.error("Hay muy pocos registros tras la limpieza. Revisa el CSV o ajusta opciones de limpieza.")
    st.stop()

# ---------- Histórico procesado ----------
st.subheader("Serie diaria (procesada)")
with st.expander("Ver primeros 30 días"):
    st.dataframe(df.head(30), use_container_width=True)

fig_hist = px.line(df, x="ds", y="y", title="Histórico de ventas (pre-procesado)")
st.plotly_chart(fig_hist, use_container_width=True)

# ---------- Prophet ----------
df_train = df.copy()
if growth_type == "logistic":
    cap_val = max(1.0, float(np.percentile(df_train["y"], 95) * 1.3))
    df_train["cap"] = cap_val
    df_train["floor"] = 0.0

# constructor correcto (con tu slider):
m = Prophet(
    yearly_seasonality=yearly_season,
    weekly_seasonality=weekly_season,
    daily_seasonality=False,
    growth=growth_type,
    changepoint_prior_scale=changepoint_scale
)

m.fit(df_train[["ds","y"] + (["cap","floor"] if growth_type == "logistic" else [])])

future = m.make_future_dataframe(periods=horizon_days, freq="D", include_history=True)
if growth_type == "logistic":
    future["cap"] = df_train["cap"].iloc[-1]
    future["floor"] = 0.0

forecast = m.predict(future)

# ---------- Plots ----------
st.subheader("Pronóstico")
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Componentes (tendencia / estacionalidades)")
fig_comp = plot_components_plotly(m, forecast)
st.plotly_chart(fig_comp, use_container_width=True)

# ---------- Descargas ----------
def to_csv_download(dfo: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    dfo.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf

# Exportar pronóstico con encabezados claros
forecast_export = forecast.rename(columns={"ds":"fecha_venta","yhat":"pronostico","yhat_lower":"lim_inf","yhat_upper":"lim_sup"})
st.download_button(
    "⬇️ Descargar pronóstico (CSV)",
    data=to_csv_download(forecast_export[["fecha_venta","pronostico","lim_inf","lim_sup"]]),
    file_name="pronostico_ventas.csv",
    mime="text/csv",
)

# Exportar histórico procesado con encabezados claros
hist_export = df.rename(columns={"ds":"fecha_venta","y":"ventas"})
st.download_button(
    "⬇️ Descargar histórico (CSV)",
    data=to_csv_download(hist_export),
    file_name="historico_ventas.csv",
    mime="text/csv",
)

# ---------- Resumen mensual ----------
st.subheader("Resumen mensual")
hist_m = df.set_index("ds")["y"].resample("MS").sum().rename("ventas_mensuales").reset_index()
pred_m = forecast.set_index("ds")["yhat"].resample("MS").sum().rename("pronostico_mensual").reset_index()

c1, c2 = st.columns(2)
with c1:
    st.write("Histórico mensual")
    st.dataframe(hist_m, use_container_width=True)
with c2:
    st.write("Pronóstico mensual")
    st.dataframe(pred_m, use_container_width=True)

fig_month = px.line(pred_m, x="ds", y="pronostico_mensual", title="Pronóstico mensual (agregado)")
st.plotly_chart(fig_month, use_container_width=True)

st.caption("Servido bajo subruta /grafica si configuras baseUrlPath en .streamlit/config.toml")

# ---------- Predicciones adicionales ----------
st.markdown("---")
st.header("🔮 Predicciones adicionales")

# Utilidad: entrena y grafica una serie ya en formato (ds,y)
def fit_and_plot(title: str, series_df: pd.DataFrame, key_suffix: str = ""):
    df_tr = series_df.copy()

    if growth_type == "logistic":
        cap_val = max(1.0, float(np.percentile(df_tr["y"], 95) * 1.3))
        df_tr["cap"] = cap_val
        df_tr["floor"] = 0.0

    model = Prophet(
        yearly_seasonality=yearly_season,
        weekly_seasonality=weekly_season,
        daily_seasonality=False,
        growth=growth_type,
        changepoint_prior_scale=changepoint_scale,
    )
    model.fit(df_tr[["ds", "y"] + (["cap", "floor"] if growth_type == "logistic" else [])])

    future = model.make_future_dataframe(periods=horizon_days, freq="D", include_history=True)
    if growth_type == "logistic":
        future["cap"] = df_tr["cap"].iloc[-1]
        future["floor"] = 0.0

    fc = model.predict(future)

    st.subheader(title)
    st.plotly_chart(plot_plotly(model, fc), use_container_width=True, key=f"fcast_{key_suffix}")
    with st.expander("Componentes", expanded=False):
        st.plotly_chart(plot_components_plotly(model, fc), use_container_width=True, key=f"comp_{key_suffix}")

# ---------- 1) Predicción por CATEGORÍA ----------
cat_col = pick_col(df_raw, ("categoria_nombre","categoria","categoria_id","category","cat"))
if cat_col:
    # metadatos de categoría por fecha normalizada
    meta_cat = df_raw[[date_col, cat_col]].copy()
    meta_cat[date_col] = pd.to_datetime(meta_cat[date_col], errors="coerce").dt.normalize()
    meta_cat = meta_cat.dropna(subset=[date_col])

    categorias = sorted(meta_cat[cat_col].astype(str).unique().tolist())
    sel_cats = st.multiselect(
        "Categorías a modelar", categorias, default=categorias[: min(3, len(categorias))]
    )

    if sel_cats:
        # Aseguramos que df (histórico) también esté normalizado al unir
        df_norm = df.copy()
        df_norm["ds"] = pd.to_datetime(df_norm["ds"]).dt.normalize()

        df_join = df_norm.merge(meta_cat.rename(columns={date_col: "ds"}), on="ds", how="left")

        for i, c in enumerate(sel_cats):
            serie_cat = df_join[df_join[cat_col].astype(str) == str(c)][["ds", "y"]]
            serie_cat = (
                serie_cat.groupby("ds", as_index=False)["y"].sum().sort_values("ds")
            )
            if fill_missing_as_zero and len(serie_cat) > 0:
                full_idx = pd.date_range(serie_cat["ds"].min(), serie_cat["ds"].max(), freq="D")
                serie_cat = (
                    serie_cat.set_index("ds")
                    .reindex(full_idx)
                    .rename_axis("ds")
                    .fillna({"y": 0.0})
                    .reset_index()
                )
            fit_and_plot(f"Categoría: {c}", serie_cat, key_suffix=f"cat_{i}")

# ---------- 2) Predicción de RECAUDACIÓN (ingresos) TOTAL ----------
rev_col = pick_col(df_raw, ("recaudacion", "ingresos", "revenue", "importe", "total"))
if not rev_col:
    price_col = pick_col(df_raw, ("precio_unitario","precio","price","unit_price"))
    if price_col:
        df_raw["_revenue_calc"] = pd.to_numeric(df_raw[value_col], errors="coerce") * pd.to_numeric(df_raw[price_col], errors="coerce")
        rev_col = "_revenue_calc"

if rev_col:
    rec = df_raw[[date_col, rev_col]].rename(columns={date_col: "ds", rev_col: "y"})
    rec["ds"] = pd.to_datetime(rec["ds"], errors="coerce").dt.normalize()
    rec["y"] = pd.to_numeric(rec["y"], errors="coerce")
    rec = rec.dropna(subset=["ds", "y"]).sort_values("ds")
    rec = rec.groupby("ds", as_index=False)["y"].sum()

    if fill_missing_as_zero and len(rec) > 0:
        full_idx = pd.date_range(rec["ds"].min(), rec["ds"].max(), freq="D")
        rec = (
            rec.set_index("ds")
            .reindex(full_idx)
            .rename_axis("ds")
            .fillna({"y": 0.0})
            .reset_index()
        )
    fit_and_plot("Recaudación total", rec, key_suffix="revenue")
else:
    st.info("No se encontró columna de recaudación ni precio para calcularla.")


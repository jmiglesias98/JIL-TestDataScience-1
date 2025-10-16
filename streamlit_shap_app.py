import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
from io import BytesIO
import requests

st.set_page_config(layout="wide", page_title="What-if SHAP Explorer")

st.title("What‑if SHAP Explorer — App Streamlit")
st.markdown("Los datos y el modelo se cargan directamente desde URLs predefinidas en GitHub.")

# CONFIGURA AQUÍ LAS URLS RAW DE GITHUB
CSV_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/clientes.csv"
MODEL_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/mejor_modelo_con_umbral_20251015.pkl"

@st.cache_data
def fetch_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

@st.cache_data
def load_df_from_bytes(bts):
    return pd.read_csv(BytesIO(bts))

@st.cache_data
def load_model_from_bytes(bts):
    return pickle.loads(bts)

try:
    csv_bytes = fetch_url(CSV_URL)
    df = load_df_from_bytes(csv_bytes)
    st.success(f"CSV cargado correctamente desde GitHub: {CSV_URL}")
except Exception as e:
    st.error(f"Error cargando CSV desde {CSV_URL}: {e}")
    st.stop()

try:
    model_bytes = fetch_url(MODEL_URL)
    model = load_model_from_bytes(model_bytes)
    st.success(f"Modelo cargado correctamente desde GitHub: {MODEL_URL}")
except Exception as e:
    st.error(f"Error cargando modelo desde {MODEL_URL}: {e}")
    st.stop()

features = df.columns.tolist()

st.sidebar.header("Configuración")

bg_size = st.sidebar.slider("Tamaño muestra background (para explainer)", min_value=10, max_value=min(500, len(df)), value=min(100, len(df)))
background = df.sample(bg_size, random_state=42)

row_selector = st.sidebar.selectbox("Selecciona cliente por índice (posición en el CSV)", options=list(range(len(df))))

base_row = df.iloc[row_selector:row_selector+1].copy()
st.write("### Cliente actual (valores)")
st.write(base_row.T)

st.write("### Controles What‑If — modifica los valores y observa el efecto")
col1, col2 = st.columns(2)
new_row = base_row.copy()

numeric_cols = new_row.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in numeric_cols]

with col1:
    st.subheader("Numéricos")
    for c in numeric_cols:
        col_min = float(df[c].quantile(0.01))
        col_max = float(df[c].quantile(0.99))
        col_val = float(base_row.iloc[0][c])
        delta = max(abs(col_val)*0.5, 1.0)
        v_min = min(col_min, col_val - delta)
        v_max = max(col_max, col_val + delta)
        step = (v_max-v_min)/100 if v_max>v_min else 1.0
        new_val = st.slider(c, min_value=v_min, max_value=v_max, value=col_val, step=step)
        new_row.at[new_row.index[0], c] = new_val

with col2:
    st.subheader("Categorías / Otros")
    for c in cat_cols:
        uniques = df[c].dropna().unique().tolist()
        try:
            default = base_row.iloc[0][c]
        except Exception:
            default = uniques[0] if len(uniques)>0 else ""
        if len(uniques) <= 20 and len(uniques) > 0:
            new_val = st.selectbox(c, options=uniques, index=uniques.index(default) if default in uniques else 0)
        else:
            new_val = st.text_input(f"{c} (valor)", value=str(default))
        new_row.at[new_row.index[0], c] = new_val

st.write("### Valores modificados")
st.write(new_row.T)

st.write("---")
st.header("Predicción y explicabilidad (SHAP)")

def make_explainer(model, background_df):
    try:
        expl = shap.Explainer(model, background_df, feature_names=background_df.columns.tolist())
        return expl
    except Exception as e:
        try:
            expl = shap.Explainer(model, background_df.sample(min(50, len(background_df)), random_state=1))
            return expl
        except Exception as e2:
            raise RuntimeError(f"No se pudo crear explainer automáticamente: {e} | {e2}")

with st.spinner("Creando explainer y calculando SHAP... esto puede tardar unos segundos"):
    explainer = make_explainer(model, background)

X_input = new_row[explainer.feature_names]

explanation = explainer(X_input)
base_value = float(explanation.base_values[0])
shap_vals = explanation.values[0]

try:
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        if proba.shape[0] >= 2:
            y_pred = proba[1]
        else:
            y_pred = proba[0]
    else:
        y_pred = model.predict(X_input)[0]
except Exception:
    y_pred = None

st.metric("Predicción (modelo)", value=str(y_pred))
st.write(f"Base value (modelo): {base_value}")

top_k = st.sidebar.slider("Número de features a mostrar (top K por impacto)", min_value=3, max_value=min(50, len(features)), value=min(10, len(features)))

feat_names = explainer.feature_names
vals = shap_vals.tolist()
order = np.argsort(np.abs(vals))[::-1]
order_top = order[:top_k]
ordered_feats = [feat_names[i] for i in order_top]
ordered_vals = [vals[i] for i in order_top]

if len(vals) > top_k:
    other_sum = float(np.sum([vals[i] for i in order[top_k:]]))
    ordered_feats.append('Otros (resto)')
    ordered_vals.append(other_sum)

x = ["Base value"] + ordered_feats + ["Prediction"]
measures = ["absolute"] + ["relative"]*len(ordered_vals) + ["total"]
y = [base_value] + ordered_vals + [None]

fig = go.Figure(go.Waterfall(
    name = "SHAP waterfall",
    orientation = "v",
    measure = measures,
    x = x,
    y = y,
    textposition = "outside",
    connector = {"line":{"color":"rgb(63, 63, 63)"}}
))
fig.update_layout(title_text=f"Waterfall de contribuciones SHAP (top {top_k})", waterfallgroupgap=0.5)

st.plotly_chart(fig, use_container_width=True)

shap_df = pd.DataFrame({"feature": ordered_feats, "shap_value": ordered_vals, "value": [X_input.iloc[0].get(f, np.nan) for f in ordered_feats]})
shap_df["abs_shap"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values("abs_shap", ascending=False)
st.subheader("Contribuciones (top features)")
st.dataframe(shap_df.reset_index(drop=True))

st.write("---")
st.info("Recuerda: pickles pueden ejecutar código arbitrario. Solo usa modelos de orígenes de confianza.")

st.write("### Exportar muestra modificada")
buf = BytesIO()
new_row.to_csv(buf, index=False)
buf.seek(0)
st.download_button("Descargar fila modificada (CSV)", data=buf, file_name="cliente_modificado.csv", mime="text/csv")

st.markdown(
    """
    **Consejos de despliegue**:\n
    - Ajusta las URLs `CSV_URL` y `MODEL_URL` al principio del script para apuntar a tus archivos en GitHub (en formato *raw*).\n
    - Añade este fichero y un `requirements.txt` con las dependencias (streamlit, pandas, shap, plotly, scikit-learn, requests) a tu repositorio.\n
    - Despliega directamente en Streamlit Cloud.
    """
)

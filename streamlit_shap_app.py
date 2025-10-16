import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import plotly.graph_objects as go
from io import BytesIO
import requests

# Importar tus clases personalizadas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === DataCleaner y PreprocesadorDinamico ===
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols = {
            "job": ["admin.", "unknown", "unemployed", "management", "housemaid",
                    "entrepreneur", "student", "blue-collar", "self-employed",
                    "retired", "technician", "services"],
            "marital": ["married","divorced","single"],
            "education": ["unknown","secondary","primary","tertiary"],
            "default": ["yes","no"],
            "housing": ["yes","no"],
            "loan": ["yes","no"],
            "contact": ["unknown","telephone","cellular"],
            "month": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
            "poutcome": ["unknown","other","failure","success"]
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col, allowed_values in self.categorical_cols.items():
            if col in df.columns:
                mode_value = df[col].mode()[0]
                df[col] = df[col].apply(
                    lambda val: mode_value if pd.isna(val)
                    else ("unknown" if val not in allowed_values and "unknown" in allowed_values
                          else (mode_value if val not in allowed_values else val))
                )
        return df

class PreprocesadorDinamico(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop_after_ohe=None):
        self.cols_to_drop_after_ohe = cols_to_drop_after_ohe
        self.ct = None

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])
        self.ct = ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ])
        self.ct.fit(X)
        return self

    def transform(self, X):
        X_t = self.ct.transform(X)
        num_cols = self.ct.transformers_[0][2]
        cat_cols = self.ct.transformers_[1][1]["ohe"].get_feature_names_out(self.ct.transformers_[1][2])
        all_cols = list(num_cols) + list(cat_cols)
        df = pd.DataFrame(X_t, columns=all_cols, index=X.index)
        if self.cols_to_drop_after_ohe:
            cols_existentes = [c for c in self.cols_to_drop_after_ohe if c in df.columns]
            df = df.drop(columns=cols_existentes, errors="ignore")
        return df

# === Streamlit app ===
st.set_page_config(layout="wide", page_title="What-if SHAP Explorer")
st.title("What-if SHAP Explorer — App Streamlit")

CSV_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/clientes.csv"
MODEL_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/modelo.joblib"

@st.cache_data
def fetch_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

@st.cache_data
def load_df_from_bytes(bts):
    return pd.read_csv(BytesIO(bts), sep=";")

@st.cache_data
def load_model_from_bytes(bts):
    return joblib.load(BytesIO(bts))

# --- Cargar CSV ---
try:
    csv_bytes = fetch_url(CSV_URL)
    df = load_df_from_bytes(csv_bytes)
    st.success(f"CSV cargado desde GitHub ({CSV_URL})")
except Exception as e:
    st.error(f"Error cargando CSV: {e}")
    st.stop()

# --- Cargar modelo ---
try:
    model_bytes = fetch_url(MODEL_URL)
    modelo_pipeline = load_model_from_bytes(model_bytes)
    st.success(f"Modelo cargado desde GitHub ({MODEL_URL})")
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

features = df.columns.tolist()

# === Sidebar: selección cliente y background ===
st.sidebar.header("Configuración")
bg_size = st.sidebar.slider("Tamaño muestra background", min_value=10, max_value=min(500, len(df)), value=min(100, len(df)))
background = df.sample(bg_size, random_state=42)

row_selector = st.sidebar.selectbox("Selecciona cliente por índice", options=list(range(len(df))))
base_row = df.iloc[row_selector:row_selector+1].copy()
st.write("### Cliente actual")
st.write(base_row.T)

# === Controles What-if ===
col1, col2 = st.columns(2)
new_row = base_row.copy()
numeric_cols = new_row.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in numeric_cols]

with col1:
    st.subheader("Numéricos")
    for c in numeric_cols:
        val = float(base_row.iloc[0][c])
        min_val, max_val = float(df[c].min()), float(df[c].max())
        step = (max_val - min_val)/100
        new_val = st.slider(c, min_value=min_val, max_value=max_val, value=val, step=step)
        new_row.at[new_row.index[0], c] = new_val

with col2:
    st.subheader("Categorías")
    for c in cat_cols:
        options = df[c].dropna().unique().tolist()
        default = base_row.iloc[0][c]
        new_val = st.selectbox(c, options=options, index=options.index(default) if default in options else 0)
        new_row.at[new_row.index[0], c] = new_val

st.write("### Valores modificados")
st.write(new_row.T)

# === SHAP Explainer ===
st.header("Predicción y SHAP")
try:
    explainer = shap.Explainer(modelo_pipeline.predict_proba, background)
    shap_values = explainer(new_row)
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
except Exception as e:
    st.error(f"Error al crear explainer: {e}")
    st.stop()

# === Predicción ===
y_prob = modelo_pipeline.predict_proba(new_row)[0]
y_pred = y_prob[1]
st.metric("Predicción (probabilidad positivo)", value=f"{y_pred:.3f}")
st.write(f"Base value (SHAP): {base_value:.3f}")

# === Waterfall plot ===
feat_names = new_row.columns.tolist()
vals = shap_values.values[0][:len(feat_names)]
order = np.argsort(np.abs(vals))[::-1]

top_k = st.sidebar.slider("Top K features", min_value=3, max_value=min(50,len(feat_names)), value=10)
ordered_feats = [feat_names[i] for i in order[:top_k]]
ordered_vals = [vals[i] for i in order[:top_k]]

fig = go.Figure(go.Waterfall(
    name="SHAP waterfall",
    orientation="v",
    measure=["absolute"] + ["relative"]*len(ordered_vals),
    x=["Base value"] + ordered_feats + ["Prediction"],
    y=[base_value] + ordered_vals + [None],
    connector={"line":{"color":"rgb(63,63,63)"}}
))
fig.update_layout(title_text=f"Waterfall SHAP (top {top_k})", waterfallgroupgap=0.5)
st.plotly_chart(fig, use_container_width=True)


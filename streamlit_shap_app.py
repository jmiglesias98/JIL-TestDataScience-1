# ============================================================
# 🔧 Imports
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from io import BytesIO
import requests
import shap
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ============================================================
# ⚙️ Configuración de la app
# ============================================================
st.set_page_config(layout="wide", page_title="What-if SHAP Explorer")
st.title("What-if SHAP Explorer — App Streamlit")
st.markdown("Los datos y el modelo se cargan directamente desde URLs predefinidas en GitHub.")

# ============================================================
# 🌐 URLs
# ============================================================
CSV_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/clientes_20251016.csv"
MODEL_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/mejor_modelo_20251016.joblib"

# ============================================================
# 🧹 Clases personalizadas
# ============================================================
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols = {
            "job": ["admin.", "unknown", "unemployed", "management", "housemaid",
                    "entrepreneur", "student", "blue-collar", "self-employed",
                    "retired", "technician", "services"],
            "marital": ["married", "divorced", "single"],
            "education": ["unknown", "secondary", "primary", "tertiary"],
            "default": ["yes", "no"],
            "housing": ["yes", "no"],
            "loan": ["yes", "no"],
            "contact": ["unknown", "telephone", "cellular"],
            "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            "poutcome": ["unknown", "other", "failure", "success"]
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
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])

        self.ct = ColumnTransformer(transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ])

        self.ct.fit(X)
        self.feature_names_out_ = self.ct.get_feature_names_out()

        if self.cols_to_drop_after_ohe:
            self.feature_names_out_ = [
                c for c in self.feature_names_out_ if c not in self.cols_to_drop_after_ohe
            ]
        return self

    def transform(self, X):
        X_t = self.ct.transform(X)
        df = pd.DataFrame(X_t, columns=self.ct.get_feature_names_out())
        if self.cols_to_drop_after_ohe:
            cols_existentes = [c for c in self.cols_to_drop_after_ohe if c in df.columns]
            df = df.drop(columns=cols_existentes, errors="ignore")
        return df.values

    def get_feature_names_out(self):
        return self.feature_names_out_

# ============================================================
# 📥 Funciones de carga
# ============================================================
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

# ============================================================
# ⚙️ Cargar datos y modelo (sin mensajes de éxito)
# ============================================================
try:
    csv_bytes = fetch_url(CSV_URL)
    df = load_df_from_bytes(csv_bytes)
except Exception as e:
    st.error(f"❌ Error cargando CSV: {e}")
    st.stop()

try:
    model_bytes = fetch_url(MODEL_URL)
    modelo_pipeline = load_model_from_bytes(model_bytes)
except Exception as e:
    st.error(f"❌ Error cargando modelo: {e}")
    st.stop()

features = df.columns.tolist()

# ============================================================
# 🛠️ Controles de configuración
# ============================================================
st.sidebar.header("⚙️ Configuración")

bg_size = st.sidebar.slider(
    "Tamaño muestra background (para explainer)",
    min_value=10, max_value=min(500, len(df)),
    value=min(100, len(df))
)
background = df.sample(bg_size, random_state=42)

row_selector = st.sidebar.selectbox(
    "Selecciona cliente por índice (posición en el CSV)",
    options=list(range(len(df)))
)
base_row = df.iloc[row_selector:row_selector+1].copy()

# ============================================================
# ✏️ Controles What-If
# ============================================================
col1, col2 = st.columns(2)
new_row = base_row.copy()
numeric_cols = new_row.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in numeric_cols]

with col1:
    st.subheader("🔢 Numéricos")
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
    st.subheader("🏷️ Categóricas / Otros")
    for c in cat_cols:
        uniques = df[c].dropna().unique().tolist()
        default = base_row.iloc[0][c] if c in base_row.columns else (uniques[0] if uniques else "")
        if len(uniques) <= 20 and len(uniques) > 0:
            new_val = st.selectbox(c, options=uniques, index=uniques.index(default) if default in uniques else 0)
        else:
            new_val = st.text_input(f"{c} (valor)", value=str(default))
        new_row.at[new_row.index[0], c] = new_val

# ============================================================
# 🧾 Tabla comparativa con resaltado de cambios
# ============================================================
comparacion = pd.DataFrame({
    "Variable": base_row.columns,
    "Valor original": base_row.iloc[0].values,
    "Valor modificado": new_row.iloc[0].values
})

def highlight_changes(row):
    orig = row["Valor original"]
    mod = row["Valor modificado"]
    if pd.isna(orig) or pd.isna(mod):
        return [""] * 3
    if isinstance(orig, (int, float)) and isinstance(mod, (int, float)):
        if mod > orig:
            return ["", "background-color: #f0f0f0", "background-color: #d0f0c0"]  # verde suave
        elif mod < orig:
            return ["", "background-color: #f0f0f0", "background-color: #f5b7b1"]  # rojo suave
    else:
        if orig != mod:
            return ["", "background-color: #f0f0f0", "background-color: #f9e79f"]  # amarillo
    return [""] * 3

st.write("### 🧾 Comparativa de valores del cliente")
st.dataframe(comparacion.style.apply(highlight_changes, axis=1), use_container_width=True)

# ============================================================
# 🧩 Predicción y SHAP — versión nativa con comparación Before/After
# ============================================================
try:
    cleaner = modelo_pipeline.named_steps["cleaner"]
    preprocessor = modelo_pipeline.named_steps["preprocessor"]
    model = modelo_pipeline.named_steps[list(modelo_pipeline.named_steps.keys())[-1]]
except KeyError:
    st.error("❌ El pipeline no contiene 'cleaner' o 'preprocessor'. Revisa los nombres de pasos.")
    st.stop()

# --- Preparar datos "antes" y "después" ---
base_row_clean = cleaner.transform(base_row)
new_row_clean = cleaner.transform(new_row)

base_row_preprocessed = preprocessor.transform(base_row_clean)
new_row_preprocessed = preprocessor.transform(new_row_clean)

X_before = base_row_preprocessed.values if hasattr(base_row_preprocessed, "values") else base_row_preprocessed
X_after = new_row_preprocessed.values if hasattr(new_row_preprocessed, "values") else new_row_preprocessed

# --- Crear background ---
background_raw = df.sample(min(bg_size, len(df)), random_state=42)
background_clean = cleaner.transform(background_raw)
background_preprocessed = preprocessor.transform(background_clean)
background_array = background_preprocessed.values if hasattr(background_preprocessed, "values") else background_preprocessed

# --- Crear explainer ---
with st.spinner("🧠 Calculando valores SHAP..."):
    explainer = shap.Explainer(model, background_array)
    shap_values_before = explainer(X_before)
    shap_values_after = explainer(X_after)

# ============================================================
# 💧 Comparación de Waterfalls
# ============================================================
st.write(f"💧 Mostrando comparativa SHAP para cliente índice **{row_selector}**")

try:
    feat_names = preprocessor.get_feature_names_out()
except Exception:
    try:
        feat_names = preprocessor.ct.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_before.shape[1])]

exp_before = shap.Explanation(
    values=shap_values_before.values[0],
    base_values=explainer.expected_value,
    data=X_before[0],
    feature_names=feat_names
)

exp_after = shap.Explanation(
    values=shap_values_after.values[0],
    base_values=explainer.expected_value,
    data=X_after[0],
    feature_names=feat_names
)

prob_before = expit(exp_before.base_values + exp_before.values.sum())
prob_after = expit(exp_after.base_values + exp_after.values.sum())

st.markdown("### 📊 Comparativa de Probabilidades")
colA, colB = st.columns(2)
colA.metric("Probabilidad original", f"{prob_before:.4f}")
colB.metric("Probabilidad modificada", f"{prob_after:.4f}",
             delta=f"{(prob_after - prob_before)*100:+.2f} pp")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Antes de modificaciones")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(exp_before, max_display=10, show=False)
    st.pyplot(fig1)

with col2:
    st.subheader("Después de modificaciones")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(exp_after, max_display=10, show=False)
    st.pyplot(fig2)

st.caption("Ambos gráficos muestran las contribuciones SHAP en log-odds. Las métricas superiores muestran las probabilidades transformadas con sigmoide.")

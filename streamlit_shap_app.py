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
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        self.feature_names_out_ = None  # 👈 para guardar los nombres finales

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

        # Guardamos nombres de características originales del ColumnTransformer
        self.feature_names_out_ = self.ct.get_feature_names_out()

        # Si hay columnas a eliminar, las quitamos de la lista
        if self.cols_to_drop_after_ohe:
            self.feature_names_out_ = [
                c for c in self.feature_names_out_
                if c not in self.cols_to_drop_after_ohe
            ]

        return self

    def transform(self, X):
        X_t = self.ct.transform(X)
        df = pd.DataFrame(X_t, columns=self.ct.get_feature_names_out())

        # Eliminar columnas después del OHE si existen
        if self.cols_to_drop_after_ohe:
            cols_existentes = [c for c in self.cols_to_drop_after_ohe if c in df.columns]
            df = df.drop(columns=cols_existentes, errors="ignore")

        return df.values  # o df si prefieres mantener nombres de columnas

    def get_feature_names_out(self):
        """Permite acceder a los nombres de las variables finales."""
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
# ⚙️ Cargar datos y modelo
# ============================================================
try:
    csv_bytes = fetch_url(CSV_URL)
    df = load_df_from_bytes(csv_bytes)
    st.success(f"✅ CSV cargado correctamente desde GitHub: {CSV_URL}")
except Exception as e:
    st.error(f"❌ Error cargando CSV: {e}")
    st.stop()

try:
    model_bytes = fetch_url(MODEL_URL)
    modelo_pipeline = load_model_from_bytes(model_bytes)
    st.success(f"✅ Modelo cargado correctamente desde GitHub: {MODEL_URL}")
except Exception as e:
    st.error(f"❌ Error cargando modelo: {e}")
    st.stop()

features = df.columns.tolist()

# ============================================================
# 🛠️ Configuración de la app
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

st.write("### Cliente actual (valores)")
st.write(base_row.T)

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

st.write("### 🧮 Valores modificados")
st.write(new_row.T)

# ============================================================
# 🧩 Predicción y SHAP — VERSIÓN FINAL USANDO PIPELINE COMPLETO
# ============================================================
# ============================================================
# 🧩 Predicción y SHAP — CORREGIDO Y ESTABLE
# ============================================================

# 1️⃣ Limpiar y transformar la fila igual que en entrenamiento
try:
    cleaner = modelo_pipeline.named_steps["cleaner"]
    preprocessor = modelo_pipeline.named_steps["preprocessor"]
    model = modelo_pipeline.named_steps[list(modelo_pipeline.named_steps.keys())[-1]]
except KeyError:
    st.error("❌ El pipeline no contiene 'cleaner' o 'preprocessor'. Revisa los nombres de pasos.")
    st.stop()

# Aplicar los mismos pasos que en el entrenamiento
new_row_clean = cleaner.transform(new_row)
new_row_preprocessed = preprocessor.transform(new_row_clean)

# Si devuelve un DataFrame con .values
if hasattr(new_row_preprocessed, "values"):
    X_input_array = new_row_preprocessed.values
else:
    X_input_array = new_row_preprocessed

# 2️⃣ Calcular la predicción
try:
    y_pred_proba = model.predict_proba(X_input_array)[0, 1]
    st.metric("Predicción (modelo)", value=str(round(y_pred_proba, 4)))
except Exception as e:
    st.error(f"❌ Error al predecir con el modelo: {e}")
    st.stop()

# 3️⃣ Crear background correctamente preprocesado
background_raw = df.sample(min(100, len(df)), random_state=42)
background_clean = cleaner.transform(background_raw)
background_preprocessed = preprocessor.transform(background_clean)
background_array = background_preprocessed.values if hasattr(background_preprocessed, "values") else background_preprocessed

# 4️⃣ Crear explainer en el espacio del modelo
with st.spinner("🧠 Calculando valores SHAP..."):
    explainer = shap.Explainer(model, background_array)
    shap_values = explainer(X_input_array)

base_value = explainer.expected_value
st.write(f"Base value: {base_value}")

# ============================================================
# 5️⃣ Mostrar gráfico Waterfall — convertido a PROBABILIDADES
# ============================================================
from scipy.special import expit  # sigmoide para pasar de log-odds → probas

vals = shap_values.values[0]

# --- 1️⃣ Nombres de features
try:
    feat_names = preprocessor.get_feature_names_out().tolist()
except Exception:
    feat_names = [f"f{i}" for i in range(len(vals))]

# --- 2️⃣ Ordenar por impacto
order = np.argsort(np.abs(vals))[::-1]
top_k = st.sidebar.slider(
    "Número de features a mostrar (top K por impacto)",
    min_value=3,
    max_value=min(50, len(feat_names)),
    value=min(10, len(feat_names))
)
ordered_feats = [feat_names[i] for i in order[:top_k]]
ordered_vals = [vals[i] for i in order[:top_k]]

# --- 3️⃣ Convertir log-odds → probabilidades paso a paso
base_logit = explainer.expected_value
base_proba = expit(base_logit)

# Probabilidades acumuladas a medida que se suman los SHAPs
prob_steps = [base_proba]
logit_current = base_logit
for v in ordered_vals:
    logit_current += v
    prob_steps.append(expit(logit_current))

# Cambios relativos de probabilidad en cada paso
prob_deltas = np.diff(prob_steps)
pred_final_proba = prob_steps[-1]

# --- 4️⃣ Construir el gráfico Waterfall en espacio de probas
x_labels = ["Base prob"] + ordered_feats + ["Predicción"]
measures = ["absolute"] + ["relative"] * len(prob_deltas) + ["total"]
y_values = [base_proba] + prob_deltas.tolist() + [None]

fig = go.Figure(go.Waterfall(
    name="SHAP (espacio de probabilidad)",
    orientation="v",
    measure=measures,
    x=x_labels,
    y=y_values,
    textposition="outside",
    connector={"line": {"color": "rgb(63, 63, 63)"}}
))
fig.update_layout(
    title_text=f"Waterfall SHAP (Top {top_k}) — espacio de probabilidad",
    waterfallgroupgap=0.5,
    yaxis_title="Probabilidad",
)
st.plotly_chart(fig, use_container_width=True)

# --- 5️⃣ Mostrar comparaciones
from math import isclose
st.metric("Probabilidad modelo", f"{y_pred_proba:.4f}")
st.metric("Probabilidad desde SHAP", f"{pred_final_proba:.4f}")
if not isclose(y_pred_proba, pred_final_proba, rel_tol=1e-2):
    st.warning("⚠️ Las probabilidades difieren ligeramente por redondeo numérico.")
st.caption("Los valores SHAP se han convertido de log-odds a probabilidades usando la función sigmoide.")

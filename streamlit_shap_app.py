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

st.set_page_config(layout="wide", page_title="What-if SHAP Explorer")

st.title("What‑if SHAP Explorer — App Streamlit")
st.markdown("Los datos y el modelo se cargan directamente desde URLs predefinidas en GitHub.")

# ============================================================
# 🌐 URLs
# ============================================================
CSV_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/clientes.csv"
MODEL_URL = "https://raw.githubusercontent.com/jmiglesias98/DataScience/refs/heads/main/modelo.joblib"

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
    st.success(f"CSV cargado correctamente desde GitHub: {CSV_URL}")
except Exception as e:
    st.error(f"Error cargando CSV: {e}")
    st.stop()

try:
    model_bytes = fetch_url(MODEL_URL)
    modelo_pipeline = load_model_from_bytes(model_bytes)
    st.success(f"Modelo cargado correctamente desde GitHub: {MODEL_URL}")
except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

features = df.columns.tolist()

# ============================================================
# 🛠️ Configuración de la app
# ============================================================
st.sidebar.header("Configuración")

bg_size = st.sidebar.slider("Tamaño muestra background (para explainer)", min_value=10, max_value=min(500, len(df)), value=min(100, len(df)))
background = df.sample(bg_size, random_state=42)

row_selector = st.sidebar.selectbox("Selecciona cliente por índice (posición en el CSV)", options=list(range(len(df))))
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
        default = base_row.iloc[0][c] if c in base_row.columns else (uniques[0] if uniques else "")
        if len(uniques) <= 20 and len(uniques) > 0:
            new_val = st.selectbox(c, options=uniques, index=uniques.index(default) if default in uniques else 0)
        else:
            new_val = st.text_input(f"{c} (valor)", value=str(default))
        new_row.at[new_row.index[0], c] = new_val

st.write("### Valores modificados")
st.write(new_row.T)

# ============================================================
# 🧩 Predicción y SHAP
# ============================================================
st.header("Predicción y explicabilidad (SHAP)")

# Transformar background y fila de entrada usando solo el preprocesador numérico
preproc = modelo_pipeline.named_steps['preprocessor']
background_num = preproc.transform(background)
new_row_num = preproc.transform(new_row)

background_df = pd.DataFrame(background_num)

def model_predict(X):
    return modelo_pipeline.named_steps['modelo'].predict_proba(X)[:, 1]

with st.spinner("Creando explainer y calculando SHAP..."):
    explainer = shap.KernelExplainer(model_predict, background_df)
    shap_values = explainer.shap_values(new_row_num, nsamples=100)
    base_value = explainer.expected_value

# ============================================================
# 🏗️ Mostrar predicción
# ============================================================
try:
    y_pred = modelo_pipeline.predict(new_row)[0]
except Exception:
    y_pred = None

st.metric("Predicción (modelo)", value=str(y_pred))
st.write(f"Base value (modelo): {base_value}")

# ============================================================
# 📊 Waterfall SHAP
# ============================================================
top_k = st.sidebar.slider("Número de features a mostrar (top K por impacto)", min_value=3, max_value=min(50, new_row_num.shape[1]), value=min(10, new_row_num.shape[1]))

vals = shap_values[0].tolist()
order = np.argsort(np.abs(vals))[::-1]
order_top = order[:top_k]
ordered_vals = [vals[i] for i in order_top]
ordered_feats = [f"Feature {i}" for i in order_top]

if len(vals) > top_k:
    other_sum = float(np.sum([vals[i] for i in order[top_k:]]))
    ordered_feats.append('Otros (resto)')
    ordered_vals.append(other_sum)

x = ["Base value"] + ordered_feats + ["Prediction"]
measures = ["absolute"] + ["relative"]*len(ordered_vals) + ["total"]
y = [base_value] + ordered_vals + [None]

fig = go.Figure(go.Waterfall(
    name="SHAP Waterfall",
    orientation="v",
    measure=measures,
    x=x,
    y=y,
    textposition="outside",
    connector={"line":{"color":"rgb(63, 63, 63)"}}
))
fig.update_layout(title_text=f"Waterfall de contribuciones SHAP (top {top_k})", waterfallgroupgap=0.5)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📋 Tabla de contribuciones
# ============================================================
shap_df = pd.DataFrame({"feature": ordered_feats, "shap_value": ordered_vals})
shap_df["abs_shap"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values("abs_shap", ascending=False)
st.subheader("Contribuciones (top features)")
st.dataframe(shap_df.reset_index(drop=True))

# ============================================================
# 💾 Exportar fila modificada
# ============================================================
from io import BytesIO
buf = BytesIO()
new_row.to_csv(buf, index=False)
buf.seek(0)
st.download_button("Descargar fila modificada (CSV)", data=buf, file_name="cliente_modificado.csv", mime="text/csv")

st.info("Pickles o joblibs pueden ejecutar código arbitrario. Solo usa modelos de confianza.")

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ============================================================
# ⚙️ Preprocesador
# ============================================================
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

        return df.values

    def get_feature_names_out(self):
        """Permite acceder a los nombres de las variables finales."""

        return self.feature_names_out_


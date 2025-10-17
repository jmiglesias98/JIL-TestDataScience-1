from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# ============================================================
# 🧹 DataCleaner
# ============================================================
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

        if "age" in df.columns:
            df["age"] = df["age"].apply(lambda x: np.nan if x < 0 else x)
            df["age"].fillna(df["age"].median(), inplace=True)

        if "balance" in df.columns:
            p05, p95 = df["balance"].quantile([0.05, 0.95])
            df["balance"] = df["balance"].clip(lower=p05, upper=p95)
            df["balance"].fillna(df["balance"].median(), inplace=True)

        if "duration" in df.columns:
            p95 = df["duration"].quantile(0.95)
            df["duration"] = df["duration"].clip(upper=p95)
            df["duration"].fillna(df["duration"].median(), inplace=True)

        if "pdays" in df.columns:
            p95 = df[df["pdays"] != -1]["pdays"].quantile(0.95)
            df["pdays"] = df["pdays"].apply(lambda x: -1 if x == -1 else min(x, p95))
            df["pdays"].fillna(df["pdays"].median(), inplace=True)

        if "previous" in df.columns:
            p95 = df["previous"].quantile(0.95)
            df["previous"] = df["previous"].clip(upper=p95)
            df["previous"].fillna(df["previous"].median(), inplace=True)

        if "campaign" in df.columns:
            df = df.drop(columns="campaign")


        return df

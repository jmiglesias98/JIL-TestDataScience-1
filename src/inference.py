import pandas as pd
import numpy as np
from .load_model import load_model

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def predict_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve las columnas originales y añade solo la probabilidad de la clase 1.
    """
    model = get_model()

    # Intentar predecir directamente con DataFrame
    try:
        probs = model.predict_proba(df)
    except Exception:
        X = df.values
        probs = model.predict_proba(X)

    # Asegurarse de que probs sea 2D
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T

    # Crear DataFrame de salida con todas las columnas originales
    out = df.copy().reset_index(drop=True)
    out["prob_class_1"] = probs[:, 1]

    return out
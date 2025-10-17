# src/inference.py
import pandas as pd
from typing import List
from .load_model import load_model
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def predict_from_df(df: pd.DataFrame):
    """
    Devuelve un DataFrame con las predicciones y probabilidades.
    Asume que el pipeline acepta un DataFrame (o numpy array con el order de features correcto).
    """
    model = get_model()

    # Si el pipeline espera un DataFrame con columnas exactas, intenta adaptarlo:
    try:
        probs = model.predict_proba(df)
        preds = model.predict(df)
    except Exception:
        # como fallback, intenta pasar los valores (si pipeline fue entrenado con arrays)
        X = df.values
        probs = model.predict_proba(X)
        preds = model.predict(X)

    # prepara salida
    out = df.copy().reset_index(drop=True)
    out["prediction"] = preds
    # si hay 2 clases, probabilidad de clase positiva en columna 1
    if probs.shape[1] == 2:
        out["prob_pos"] = probs[:, 1]
    else:
        # si varias clases, añade columnas por clase
        for i in range(probs.shape[1]):
            out[f"prob_class_{i}"] = probs[:, i]

    return out

def predict_from_csv(path: str):
    df = pd.read_csv(path, sep=None, engine='python')  # intenta autodetectar separador
    return predict_from_df(df)

import joblib
import glob
import os
from typing import Any

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def find_latest_model(pattern="mejor_modelo_*.joblib"):
    paths = glob.glob(os.path.join(MODEL_DIR, pattern))
    if not paths:
        raise FileNotFoundError(f"No se encontró ningún modelo con patrón {pattern} en {MODEL_DIR}")
    # ordena por fecha de modificación y devuelve el más reciente
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def load_model(path: str = None) -> Any:
    if path is None:
        path = find_latest_model()
    model = joblib.load(path)
    return model

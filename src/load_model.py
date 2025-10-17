# src/load_model.py
import joblib
import glob
import os
from typing import Any

# importa las clases reales desde tu paquete
from .data_cleaner import DataCleaner
from .preprocesador_dinamico import PreprocesadorDinamico

# añadimos import de __main__ para inyectar las clases que el pickle espera
import __main__

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def find_latest_model(pattern="mejor_modelo_*.joblib"):
    paths = glob.glob(os.path.join(MODEL_DIR, pattern))
    if not paths:
        raise FileNotFoundError(f"No se encontró ningún modelo con patrón {pattern} en {MODEL_DIR}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def load_model(path: str = None) -> Any:
    if path is None:
        path = find_latest_model()

    # ===== INYECCIÓN: asegura que __main__.DataCleaner (y PreprocesadorDinamico) existen =====
    # El pickle busca __main__.DataCleaner, así que la ponemos ahí antes de cargar.
    setattr(__main__, "DataCleaner", DataCleaner)
    setattr(__main__, "PreprocesadorDinamico", PreprocesadorDinamico)
    # ========================================================================================

    model = joblib.load(path)
    return model
# src/load_model.py
import joblib
import glob
import os
from typing import Any

# importa las clases reales desde tu paquete
from .data_cleaner import DataCleaner
from .preprocesador_dinamico import PreprocesadorDinamico

# añadimos import de __main__ para inyectar las clases que el pickle espera
import __main__

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

def find_latest_model(pattern="mejor_modelo_*.joblib"):
    paths = glob.glob(os.path.join(MODEL_DIR, pattern))
    if not paths:
        raise FileNotFoundError(f"No se encontró ningún modelo con patrón {pattern} en {MODEL_DIR}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def load_model(path: str = None) -> Any:
    if path is None:
        path = find_latest_model()

    # ===== INYECCIÓN: asegura que __main__.DataCleaner (y PreprocesadorDinamico) existen =====
    # El pickle busca __main__.DataCleaner, así que la ponemos ahí antes de cargar.
    setattr(__main__, "DataCleaner", DataCleaner)
    setattr(__main__, "PreprocesadorDinamico", PreprocesadorDinamico)
    # ========================================================================================

    model = joblib.load(path)
    return model

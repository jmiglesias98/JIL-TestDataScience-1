# src/app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from .inference import predict_from_df, get_model
import uvicorn

app = FastAPI(title="Inference API")

class Rows(BaseModel):
    rows: list  # lista de dicts

@app.on_event("startup")
def startup():
    # precarga modelo para que la primera petición sea rápida
    try:
        get_model()
    except Exception as e:
        # no falla el arranque, pero lo registramos
        print("Advertencia: no se pudo cargar modelo en startup:", e)

@app.post("/predict")
def predict_endpoint(payload: Rows):
    try:
        df = pd.DataFrame(payload.rows)
        out = predict_from_df(df)
        # devolver sólo predicción y prob si prefieres, aquí devolvemos todo
        return {"predictions": out.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Recibe un CSV con las mismas columnas que usaste en entrenamiento.
    """
    try:
        contents = await file.read()
        s = contents.decode("utf-8")
        df = pd.read_csv(StringIO(s))
        out = predict_from_df(df)
        return {"predictions": out.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ejecutable localmente con: uvicorn src.app:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

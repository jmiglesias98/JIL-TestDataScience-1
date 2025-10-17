# src/app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from io import StringIO, BytesIO
from fastapi.responses import StreamingResponse
from .inference import predict_from_df
from .load_model import load_model
import uvicorn

app = FastAPI(title="Inference API")

@app.get("/")
def read_root():
    return {"message": "¡Hola, FastAPI funciona!"}

@app.on_event("startup")
def startup():
    try:
        global MODEL
        MODEL = load_model()
    except Exception as e:
        print("Advertencia: no se pudo cargar el modelo en startup:", e)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Recibe un CSV y devuelve directamente un CSV con predicciones y probabilidades.
    """
    try:
        contents = await file.read()
        s = contents.decode("utf-8")
        df = pd.read_csv(StringIO(s), sep=';')  # punto y coma
        out = predict_from_df(df)

        # Convertir DataFrame a CSV en memoria
        stream = BytesIO()
        out.to_csv(stream, index=False, sep = ';')
        stream.seek(0)

        return StreamingResponse(stream, media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=predicciones.csv"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

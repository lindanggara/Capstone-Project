from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

model = joblib.load('safety_model.joblib')

app = FastAPI(
    title="Safety Incident NLP API",
    description="API untuk klasifikasi laporan kecelakaan kerja",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "message": "Safety Incident NLP API Running"
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        prediction = model.predict([request.text])

        return {
            "success": True,
            "input_text": request.text,
            "prediction": prediction[0]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
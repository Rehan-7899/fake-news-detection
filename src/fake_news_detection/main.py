from fastapi import FastAPI , HTTPException
import joblib
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str=Field(..., min_length=10)
    
class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

from fake_news_detection.preprocessing import preprocessing_data

app=FastAPI()

model = joblib.load("src/fake_news_detection/model/model.joblib")

tf_idf = joblib.load("src/fake_news_detection/model/tfidf.joblib")

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tf_idf_loaded": tf_idf is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    processed_text = preprocessing_data(request.text)
    
    
    if not processed_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text contain no usable content."
        ) 
    
    
    X = tf_idf.transform([processed_text])
    
    prediction = model.predict(X)[0].item()
   
    
    probabilities = model.predict_proba(X)[0].tolist()
    
    if prediction == 0:
        result = "Fake"
    else:
        result = "Real"
    
    return {
        "prediction": result,
        "probabilities": {
            "fake": float(f"{(probabilities[0])*100:.3f}"),
            "real": float(f"{(probabilities[1])*100:.3f}")
            }
    }
from fastapi import FastAPI , HTTPException
import joblib
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str=Field(..., min_length=10)


from fake_news_detection.preprocessing import preprocessing_data

app=FastAPI()

model = joblib.load("src/fake_news_detection/model/model.joblib")

tf_idf = joblib.load("src/fake_news_detection/model/tfidf.joblib")

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running."}

@app.post("/predict")
def predict(request: PredictRequest):
    processed_text = preprocessing_data(request.text)
    
    try:
        X = tf_idf.transform([processed_text])
    
    except  Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Text contain no usable content."
        ) 
    
    prediction = model.predict(X)[0].item()
   
    
    probabilities = model.predict_proba(X)[0].tolist()
    
    if prediction == 0:
        result = "Fake"
    else:
        result = "True"
    
    return {
        "prediction": result,
        "probabilities": {
            "fake": probabilities[0],
            "real": probabilities[1]}
    }
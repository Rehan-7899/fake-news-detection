from fastapi import FastAPI 
import joblib

from fake_news_detection.preprocessing import preprocessing_data

app=FastAPI()

model = joblib.load("src/fake_news_detection/model/model.joblib")

tf_idf = joblib.load("src/fake_news_detection/model/tfidf.joblib")

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running."}
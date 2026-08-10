import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fake_news_detection.preprocessing import preprocessing_data


df=pd.read_csv("data/news.csv")


#Preprocess
df['cleaned_text'] = df['text'].apply(preprocessing_data)


#Split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)


#Tf - Idf
tf_idf = TfidfVectorizer(min_df=5)

X_train_tfidf = tf_idf.fit_transform(X_train)
X_train_tfidf = tf_idf.transform(X_test)

#Train Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)


#Save
joblib.dump(lr_model, "src/fake_news_detection/model/model.joblib")
joblib.dump(tf_idf, "src/fake_news_detection/model/tfidf.joblib")

print("Model and Tf-Idf saved.")
import pandas as pd
import regex as re
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

df=pd.read_csv("data/news.csv")

def data_quality_check(df):
    quality_report = {
        'total_records': len(df),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values' : df.isnull().sum().to_dict(),
        'description' : df.describe(include="all")
    }
    
    return quality_report

quality_report= data_quality_check(df)
#print(quality_report)



def preprocessing_data(text):
    #Remove HTML Tags
    text=re.sub(r"<.*?>", "", text)
        
    #Remove URL
    text=re.sub(r"http\S+|www\.\S+", "", text)
    
    #Text Lowercase
    text=text.lower()
    
    #Remove Punctuations(unicode also) 
    text=re.sub(r"\p{P}", " ", text)
    
    #Remove Numbers
    text=re.sub(r"\d+", "", text)
    
    #Remove Multiple Whitespaces
    text=re.sub(r"\s+", " ", text).strip()
    
    #Remove Stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)
    
df["cleaned_text"] = df["text"].apply(preprocessing_data)


df['word_count']=df['cleaned_text'].apply(lambda x: len(x.split()))
df['char_count']=df['cleaned_text'].apply(len)

#print(df['word_count'].describe())
#print(df['char_count'].describe())

#top 20 most frequent words
def most_freq_words(series, n):
    all_words = " ".join(series).split()
    
    word_freq = Counter(all_words)
    
    return word_freq.most_common(n)

top_20 = most_freq_words(df['cleaned_text'], 20)


X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size = 0.2,
    random_state = 42,
    stratify = df['label']
)

tf_idf = TfidfVectorizer(min_df=5)
X_train_tfidf =tf_idf.fit_transform(X_train)
X_test_tfidf =tf_idf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred = lr_model.predict(X_test_tfidf)

print("Accuracy: ", accuracy_score(y_test, y_pred)*100)
print("Classfication Report: ", classification_report(y_test, y_pred))


cm=confusion_matrix(y_test, y_pred)
print(cm)

disp= ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

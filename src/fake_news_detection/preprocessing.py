import nltk
import regex as re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

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
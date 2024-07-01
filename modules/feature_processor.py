import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop_words = stopwords.words('english')

tfidf = TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1, 2))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def extract_features(df):
    X = tfidf.fit_transform(df['Text']).toarray()
    y = df['language']
    return X, y

def extract_features_for_input(text):
    X_input = tfidf.transform([text]).toarray()
    return X_input
from sklearn.feature_extraction.text import CountVectorizer
from data import make_dataset


def make_features(df, vectorizer = None):
    y = df["is_comic"]
    
    if vectorizer is None:
        vectorizer = CountVectorizer(analyzer='word')
        X = vectorizer.fit_transform(df["video_name"])
    else:
        X = vectorizer.transform(df["video_name"])
    
    return X, y, vectorizer

 def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("french"))
    words = text.lower().split()
    return " ".join([stemmer.stem(word) for word in words if word not in stop_words])
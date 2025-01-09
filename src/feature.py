from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import re


def make_features(df):
    y = df["is_comic"]
    X = df["video_name"].apply(clean_features)
    return X, y

def clean_features(str_input):
    stemmer = FrenchStemmer()
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [word for word in words if word not in stopwords.words("French")]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)
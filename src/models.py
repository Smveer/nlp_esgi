from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


def make_model():
    return Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("model", RandomForestClassifier()),
        ]
    )

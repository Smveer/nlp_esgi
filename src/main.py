import click
import joblib
import pandas as pd
from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y, vectorizer = make_features(df)

    model = make_model()
    model.fit(X, y)

    joblib.dump(model, model_dump_filename)
    joblib.dump(vectorizer, "models/vectorizer.json")

    

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    
    model = joblib.load(model_dump_filename)
    vectorizer = joblib.load("models/vectorizer.json")
    X, y, vectorizer = make_features(df, vectorizer = vectorizer)

    
    predictions = model.predict(X)
    
    results_df = pd.DataFrame({
        "video_name": X,
        "is_comic_pred": predictions
    }).to_csv(output_filename, index=False)

    return model.predict(X)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, vectorizer = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Run k-fold cross validation. Print results
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

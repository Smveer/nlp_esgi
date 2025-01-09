import click
import joblib
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass

@click.command()
@click.option("--key", default="1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs", help="Google sheet file key")
@click.option("--id", default="1482158622", help="Google sheet file sheet id")
@click.option("--path_to", default="data/raw/train.csv", help="Where to download the sheet")
def download_google_sheet_as_csv(key, id, path_to):
    url = "https://docs.google.com/spreadsheets/d/" + key + "/gviz/tq?tqx=out:csv&sheet=" + id
    output_file = path_to
    response = requests.get(url)
    with open(output_file, 'wb') as file:
        file.write(response.content)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
    model.fit(X, y)

    return joblib.dump(model, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, _ = make_features(df)

    model = joblib.load(model_dump_filename)
    predicted = model.predict(X)
    pd.DataFrame(predicted).to_csv(output_filename, index=False)

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    accuracies = cross_val_score(model, X, y, cv=10, scoring="accuracy")

    print(np.mean(accuracies))
    return accuracies


cli.add_command(download_google_sheet_as_csv)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

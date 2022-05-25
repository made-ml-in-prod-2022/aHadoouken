from email.policy import default
import requests
import click
import pandas as pd


@click.command()
@click.option("--data_path", required=True, type=str, help="path to data")
@click.option("--model", default="lr", type=str, help="lr/rf")
@click.option("--host", default="0.0.0.0", help="Host ip")
@click.option("--port", default=8000, help="Host port")
def main(data_path, model, host, port):
    if model == "lr":
        model_name = "logistic_regression"
    elif model == "rf":
        model_name = "random_forest"
    else:
        raise ValueError(f"Invalid model type {model}")
    df = pd.read_csv(data_path)
    df.drop(["condition"], axis=1, inplace=True)
    json = {
        "data": df.values.tolist(),
        "features_names": df.columns.to_list(),
        "model": model_name,
    }
    print(json)
    response = requests.post(
        f"http://{host}:{port}/predict",
        json=json,
    )
    print(f"Status code:\n{response.status_code}")
    print(f"Result:\n{response.json()}")


if __name__ == "__main__":
    main()

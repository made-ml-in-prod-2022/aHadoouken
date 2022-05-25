from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

valid_json_pred = {
    "data": [
        [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
        [69, 0, 0, 140, 239, 0, 0, 151, 0, 1.8, 0, 2, 0],
        ],
        "features_names": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        ],
        "model": "logistic_regression",
        }


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry point of our predictor"


def test_healtz():
    with TestClient(app) as client:
        response = client.get("/healz")
        assert response.status_code == 200


def test_predict():
    with TestClient(app) as client:
        response = client.get("/predict")
        assert response.status_code == 405

        response = client.post("/predict", json={})
        assert response.status_code == 422

        non_valid_json = valid_json_pred.copy()
        non_valid_json["model"] = "wrong_model"
        response = client.post("/predict", json=non_valid_json)
        assert response.status_code == 422

        non_valid_json = valid_json_pred.copy()
        non_valid_json["data"] = [["wtf", 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
                                  [69, 0, 0, 140, 239, 0, 0, 151, 0, 1.8, 0, 2, 0]]
        response = client.post("/predict", json=non_valid_json)
        assert response.status_code == 422

        non_valid_json = valid_json_pred.copy()
        non_valid_json["data"] = [[1, 2, 3]]
        response = client.post("/predict", json=non_valid_json)
        assert response.status_code == 422

        non_valid_json = valid_json_pred.copy()
        non_valid_json["features_names"] = ["not", "enough", "features"]
        response = client.post("/predict", json=non_valid_json)
        assert response.status_code == 422

        response = client.post(
            "/predict",
            json=valid_json_pred,
        )
        assert response.status_code == 200

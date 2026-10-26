from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction():
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "predicted_class" in result
    assert "class_name" in result

    assert result["class_name"] in ["setosa", "versicolor", "virginica"]
    assert result["prediction"] in [0, 1, 2]
    


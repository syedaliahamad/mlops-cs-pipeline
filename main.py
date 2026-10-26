from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field

app = FastAPI(title='Iris Model API')
model_path = Path("models/iris_model.joblib")
model=joblib.load(model_path)

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: IrisRequest):
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]
    prediction = int(model.predict(features)[0])
    return {
        "prediction": prediction,
        "class_name": ["setosa", "versicolor", "virginica"][prediction]
    }


import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

model_file = "model_C=1.0.bin"

# Load model & DictVectorizer
with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Churn Prediction API")


# ---- Request Schema ----
class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float


# ---- Prediction Endpoint ----
@app.post("/predict")
def predict(customer: Customer):
    X = dv.transform([customer.model_dump()])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    return {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }

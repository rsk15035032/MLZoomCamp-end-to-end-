import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

model_file = "model.bin"

# Load model & DictVectorizer
with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Credit Scoring API")


# ---- Request Schema ----
class Customer(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


# ---- Prediction Endpoint ----
@app.post("/predict")
def predict(customer:Customer):
    X = dv.transform([customer.model_dump()])
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    proba = model.predict(dmatrix)[0]
    if proba >= 0.5:
        return {
                "probability": float(proba),
                 "Status": 'Default'
                }
    else:
        return {
                "probability": float(proba),
                 "Status": 'Ok'
                }
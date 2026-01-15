import pickle
from fastapi import FastAPI
from pydantic import BaseModel


# Create FastAPI app
app = FastAPI(title="Churn Prediction API")


# Health check
@app.get("/")
def health():
    return {"status": "API is running"}


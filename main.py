from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

pipeline = joblib.load("customer_churn_pipeline.pkl")

class Custmor(BaseModel):
    age: float
    tenure_months : float
    monthly_charges : float
    support_calls : float   
    contract_months : float


@app.post('/prediction')
def predict_churn(custmer : Custmor):
    data = pd.DataFrame({
        "age" : [custmer.age],
        "tenure_months" : [custmer.tenure_months],
        "monthly_charges" : [custmer.monthly_charges],
        "support_calls" : [custmer.support_calls],
        "contract_months" : [custmer.contract_months]
    })

    prediction = pipeline.predict(data)[0]
    probablity = pipeline.predict_proba(data)[0][1]

    return {
        "prediction" : int(prediction),
        "churn_probablity" : float(probablity)
    }



#import libraries
from fastapi import FastAPI, Query
import uvicorn
import joblib
import pandas as pd

#create app object
app = FastAPI()

###create home
@app.get('/')
def home():
    return{'message':'Welcome to the Sepsis Prediction Model'}


##load the model
model = joblib.load("model\rf_model.joblib")

# Endpoint to get predictions
@app.post("/predict")
def predict_sepsis(
    Plasma_glucose: int = Query(..., description="Plasma_glucose"),
    PL: int = Query(..., description="Blood_work_result1"),
    Blood_Pressure: int = Query(..., description="Blood_Pressure"),
    SK: int = Query(..., description="Blood_work_result2"),
    TS: int = Query(..., description="Blood_work_result3"),
    M11: float = Query(..., description="BMI"),
    BD2: float = Query(..., description="Blood_work_result4"),
    Age: int = Query(..., description="Age")
   
):
    # Convert input data to the format expected by the model
    input_data = pd.DataFrame([{
        "Plasma_glucose": Plasma_glucose,
        "Blood_Work_R1": PL,
        "Blood_Pressure": Blood_Pressure,
        "Blood_Work_R2": SK,
        "Blood_Work_R3": TS,
        "BMI": M11,
        "Blood_Work_R4": BD2,
        "Age": Age
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    sepsis_status = "Patient is susceptible to sepsis" if prediction == 1 else "Patient is not susceptible to sepsis"

    # Return the prediction
    return {"prediction": sepsis_status}
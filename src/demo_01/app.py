from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import uvicorn
import numpy as np

# Load the model
model = joblib.load("model/rf_model.joblib")

# Create the FastAPI app object
app = FastAPI(title="Sepsis Prediction API")

# Define the input data model
class SepsisInput(BaseModel):
    Plasma_glucose: int
    Blood_work_results1: int
    Blood_Pressure: int
    Blood_work_results2: int
    Blood_work_results3: int
    Body_mass_Index: float
    Blood_work_results4: float
    Age: int

# Status endpoint : check if the api is online
@app.get("/status")
async def status():
    return {"message": "online"}

# the root endpoint
@app.get("/") 
async def root():
    return {"info": "Sepsis Prediction API: This API application serves as a bridge between the model and the user" }

# the home endpoint
@app.get('/home')
def home():
    return {'message': 'Welcome to the Sepsis Prediction Model'}

# the prediction endpoint
@app.post("/predict")
async def predict_sepsis(sepsis:SepsisInput,Plasma_glucose: int, Blood_work_results1: int, Blood_Pressure: int, Blood_work_results2: int, Blood_work_results3: int, Body_mass_Index: float, Blood_work_results4: float, Age: int):
    try:
        # Map the input feature names to the model's expected feature names
        input_data = {
            "PRG": sepsis.Plasma_glucose,
            "PL": sepsis.Blood_work_results1,
            "PR": sepsis.Blood_Pressure,
            "SK": sepsis.Blood_work_results2,
            "TS": sepsis.Blood_work_results3,
            "M11": sepsis.Body_mass_Index,
            "BD2": sepsis.Blood_work_results4,
            "Age": sepsis.Age
        }

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make a prediction
        prediction = model.predict(input_df)[0]
        sepsis_status = "Patient is susceptible to sepsis" if prediction == 1 else "Patient is not susceptible to sepsis"

        # Calculate confidence score
        confidence_score = model.predict_proba(input_df)[0]
        probability = confidence_score[1]  # Probability of positive sepsis prediction

       # Define status icon and explanation based on the prediction
        if prediction == 1:
            status_icon = "✘"  # Red 'X' icon for positive sepsis prediction
            sepsis_explanation = "A high prediction for sepsis indicates that the patient may be experiencing severe symptoms due to an infection and requires immediate medical attention."
        else:
            status_icon = "✔"  # Green checkmark icon for negative sepsis prediction
            sepsis_explanation = "A low prediction for sepsis suggests that the patient is not currently showing symptoms of the condition."

        # Construct the statement
        statement = f"{status_icon} {sepsis_status} with a probability of {probability:.2f}. {sepsis_explanation}"

        # Construct the result
        result = {
            'predicted_sepsis': sepsis_status,
            'statement': statement,
            'execution_msg': "The execution went fine",
            'execution_code': 1
        }

        # Return the result
        return result

    except Exception as e:
        print(f"Something went wrong during the sepsis prediction: {e}")
        return {"execution_msg": "The execution went wrong", "execution_code": 0, "prediction": None}

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)

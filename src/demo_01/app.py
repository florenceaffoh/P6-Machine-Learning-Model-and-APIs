from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import uvicorn

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

# Define the root endpoint
@app.get("/") 
async def root():
    return {"info": "Sepsis Prediction API: This API application serves as a bridge between the model and the user" }

# Define the home endpoint
@app.get('/home')
def home():
    return {'message': 'Welcome to the Sepsis Prediction Model'}

# Define the prediction endpoint
@app.post("/predict")
async def predict_sepsis(sepsis: SepsisInput):
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
        confidence_score = model.predict_proba(input_df).max()

        # Return the prediction and confidence score
        return {"prediction": sepsis_status, "confidence_score": confidence_score, "execution_msg": "The execution went fine", "execution_code": 1}

    except Exception as e:
        print(f"Something went wrong during the sepsis prediction: {e}")
        return {"execution_msg": "The execution went wrong", "execution_code": 0, "prediction": None}

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)

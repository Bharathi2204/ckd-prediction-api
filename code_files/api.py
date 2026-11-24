from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# STEP 1: Home Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to CKD Prediction API"}

# STEP 2: Patient Input Features
class PatientData(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wbcc: float
    rbcc: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str

# STEP 3: Load Trained ML Model
model = joblib.load(
    r"D:\Datascience\projects\healthcare\project_1_chronic_kidney_disease\models\ckd_median_mode_svm_pipeline.joblib"
)

# # STEP 4: Predict CKD
# @app.post("/predict")
# def predict(data: PatientData):

#     patient_data = pd.DataFrame([data.model_dump()])

#     prediction = model.predict(patient_data)[0]

#     result = "CKD" if prediction == 1 else "Not CKD"

#     return {"prediction": result}
# STEP 7: Error Handling
from fastapi import HTTPException

@app.post("/predict")
def predict(data: PatientData):
    try:
        patient_data = pd.DataFrame([data.model_dump()])
        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0]
        confidence = float(max(probability)) * 100
        result = "CKD" if prediction == 1 else "Not CKD"

        return {
        "prediction": result,
        "confidence": round(confidence, 2)
         }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn

class Patient(BaseModel):
    age: int = Field(..., ge=0)
    gender: Literal["f", "m"]
    smoking: Literal["no", "yes"]
    hx_smoking: Literal["no", "yes"]
    hx_radiothreapy: Literal["no", "yes"]
    thyroid_function: Literal[
        "euthyroid",
        "clinical_hyperthyroidism",
        "clinical_hypothyroidism",
        "subclinical_hyperthyroidism",
        "subclinical_hypothyroidism"
    ]
    physical_examination: Literal[
        "single_nodular_goiter-left",
        "multinodular_goiter",
        "single_nodular_goiter-right",
        "normal",
        "diffuse_goiter"
    ]
    adenopathy: Literal[
        "no",
        "right",
        "extensive",
        "left",
        "bilateral",
        "posterior"
    ]
    pathology: Literal[
        "micropapillary",
        "papillary",
        "follicular",
        "hurthel_cell"
    ]
    focality: Literal["uni-focal", "multi-focal"]
    tumor: Literal["t1a", "t1b", "t2", "t3a", "t3b", "t4a", "t4b"]
    node: Literal["n0", "n1b", "n1a"]
    metastasis: Literal["m0", "m1"]
    stage: Literal["i", "ii", "ivb", "iii", "iva"]
    response: Literal[
        "indeterminate",
        "excellent",
        "structural_incomplete",
        "biochemical_incomplete"
    ]

class PredictResponse(BaseModel):
    recurred_probability: float
    recurred: bool
    
app = FastAPI(title="thyroid-recurrence-prediction")

with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict_single(patient_dict):
    """
    Takes a dict of patient features and returns predicted probability of recurrence.
    """
    prob = pipeline.predict_proba([patient_dict])[0, 1]
    return float(prob)


@app.post("/predict")
def predict(patient: Patient) -> PredictResponse:
    prob = predict_single(patient.model_dump())

    return PredictResponse(
        recurred_probability=prob,
        recurred=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
"""AutismCare Diagnosis Age API.

Predicts the age at which a child is diagnosed with autism.
Endpoints: /predict, /retrain and /docs.
"""

import io
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

MODEL_PATH = Path(__file__).parent / "best_model.pkl"
TARGET = "diagnosis_age"
FEATURES = [
    "income_poverty_ratio", "severity", "adhd", "anxiety", "developmental_delay", "speech_disorder",
    "intellectual_disability", "sex", "insurance_type", "home_language",
]

YesNo = Literal["yes", "no"]


# --- Input and output formats ---

class ChildInput(BaseModel):
    """The 10 inputs for one child. All are required."""

    # strict=True: reject wrong types, e.g. "150" as text
    # extra="forbid": reject unknown fields
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "income_poverty_ratio": 180,
                "severity": "mild",
                "adhd": "yes",
                "anxiety": "yes",
                "developmental_delay": "no",
                "speech_disorder": "no",
                "intellectual_disability": "no",
                "sex": "female",
                "insurance_type": "private",
                "home_language": "english",
            }
        },
    )

    income_poverty_ratio: float = Field(
        ge=50, le=400, allow_inf_nan=False,
        description="Family income as a percentage of the US federal poverty line (50 = 50% or less, 400 = 400% or more)",
    )
    severity: Literal["mild", "moderate", "severe"] = Field(description="Autism severity as described by the parent")
    adhd: YesNo = Field(description="Has a doctor ever said the child has ADHD?")
    anxiety: YesNo = Field(description="Has a doctor ever said the child has anxiety problems?")
    developmental_delay: YesNo = Field(description="Has a doctor ever said the child has a developmental delay?")
    speech_disorder: YesNo = Field(description="Has a doctor ever said the child has a speech or other language disorder?")
    intellectual_disability: YesNo = Field(description="Has a doctor ever said the child has an intellectual disability?")
    sex: Literal["male", "female"] = Field(description="Child's sex")
    insurance_type: Literal["public", "private", "public_and_private", "uninsured"] = Field(
        description="Child's health insurance type"
    )
    home_language: Literal["english", "spanish", "other"] = Field(description="Main language spoken at home")


class TrainingRow(ChildInput):
    """One CSV row: the 10 inputs plus the diagnosis age."""

    # CSV values are text, so allow conversion. Ignore extra columns.
    model_config = ConfigDict(strict=False, extra="ignore")
    diagnosis_age: float = Field(ge=0, le=17, allow_inf_nan=False, description="Age when first told the child had autism (years)")


class PredictionOutput(BaseModel):
    predicted_diagnosis_age_years: float
    message: str


class RetrainOutput(BaseModel):
    message: str
    rows_received: int
    training_rows_before: int
    training_rows_after: int
    test_mse_before: float
    test_mse_after: float
    test_rmse_after_years: float
    retrained_at: str


# --- Model ---

def load_model(path=MODEL_PATH):
    """Load the saved model (preprocessing + random forest)."""
    return joblib.load(path)


def predict_diagnosis_age(child, model):
    """Predict the diagnosis age for one child (same as the notebook)."""
    row = pd.DataFrame([child]) if isinstance(child, dict) else child
    return float(model.predict(row[FEATURES])[0])


model = load_model()           # load the model once at startup
model_lock = threading.Lock()  # one request at a time can use the model


# --- App and CORS ---

app = FastAPI(
    title="AutismCare Diagnosis Age API",
    description=(
        "Predicts the age (in years) at which a child is likely to be diagnosed with autism, from the child's "
        "co-occurring conditions, autism severity and family characteristics. Outreach teams use it to contact "
        "families at risk of a late diagnosis first. Trained on the US National Survey of Children's Health (2022 and 2024)."
    ),
    version="1.0.0",
)

# CORS: which websites can call this API from a browser.
# Allowed: our Render site and the local Flutter web page, with GET and POST only.
# Blocked: all other websites (no "*"), other methods and cookies, to keep the API safe.
# The mobile app is not a browser, so CORS does not affect it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://autism-diagnosis-age-api.onrender.com",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


# --- Endpoints ---

@app.get("/")
def home():
    return {
        "service": "AutismCare Diagnosis Age API",
        "predict": "POST /predict",
        "retrain": "POST /retrain (CSV upload)",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(child: ChildInput):
    with model_lock:
        age = predict_diagnosis_age(child.model_dump(), model)
    age = round(age, 1)
    return PredictionOutput(
        predicted_diagnosis_age_years=age,
        message=f"This child is predicted to be diagnosed with autism at about {age} years old.",
    )


@app.post("/retrain", response_model=RetrainOutput)
async def retrain(file: UploadFile = File(description="CSV with the 10 input columns plus diagnosis_age")):
    """Upload a CSV of new children. The model retrains on old + new data, and /predict uses it right away."""
    global model

    # Check the file
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        new_data = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read the CSV file: {error}")

    missing = [column for column in FEATURES + [TARGET] if column not in new_data.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"The CSV is missing these columns: {missing}")
    if new_data.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")

    # Check each row with the /predict rules (empty cells count as missing)
    records = new_data[FEATURES + [TARGET]].astype(object).where(new_data[FEATURES + [TARGET]].notna(), None)
    try:
        TypeAdapter(list[TrainingRow]).validate_python(records.to_dict(orient="records"))
    except ValidationError as error:
        # Show up to 10 errors. "+ 2" makes row numbers match the CSV file.
        problems = [
            f"row {err['loc'][0] + 2}, column '{err['loc'][1]}': {err['msg']}" for err in error.errors()[:10]
        ]
        raise HTTPException(status_code=422, detail={"message": "Some rows are invalid.", "problems": problems})

    new_rows = new_data[FEATURES + [TARGET]].copy()
    new_rows["income_poverty_ratio"] = new_rows["income_poverty_ratio"].astype(float)
    new_rows[TARGET] = new_rows[TARGET].astype(float)

    with model_lock:
        old_model = model
        training_data = pd.concat([old_model.training_data_, new_rows], ignore_index=True)
        test_data = old_model.test_data_

        # Retrain on the old data plus the new rows
        new_model = clone(old_model).fit(training_data[FEATURES], training_data[TARGET])
        new_model.training_data_ = training_data
        new_model.test_data_ = test_data

        # Compare the old and new model on the same test set
        mse_before = mean_squared_error(test_data[TARGET], old_model.predict(test_data[FEATURES]))
        mse_after = mean_squared_error(test_data[TARGET], new_model.predict(test_data[FEATURES]))

        # Use the new model now. A restart reloads best_model.pkl.
        model = new_model

    return RetrainOutput(
        message="Model retrained on the original training data plus the uploaded rows, and is now in use.",
        rows_received=len(new_rows),
        training_rows_before=len(old_model.training_data_),
        training_rows_after=len(training_data),
        test_mse_before=round(float(mse_before), 4),
        test_mse_after=round(float(mse_after), 4),
        test_rmse_after_years=round(float(np.sqrt(mse_after)), 3),
        retrained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

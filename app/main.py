"""Prediction service for the Kochi Student Housing Board.

    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"prediction": <fair monthly rent in Rs>}

The client's two rules live in the request schema (pydantic Field), so a
bad request gets status 422 and never reaches the model.

Run locally:  uvicorn app.main:app --port 8000
"""

import os
import time
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.json_logging import get_logger

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")

# one JSON line per event, to stdout (see app/json_logging.py)
logger = get_logger()


# ---------------------------------------------------------------------------
# Request schema: exactly the 8 fields from PROBLEM.md section 5.
# ---------------------------------------------------------------------------
class RentRequest(BaseModel):
    distance_to_campus_km: float = Field(..., ge=0, description="km to campus")
    room_area_sqft: float = Field(..., gt=0, le=650, description="rule 1: above 650 sq ft is impossible")
    sharing: int = Field(..., ge=1, description="students sharing the room")
    meals_per_day: int = Field(..., ge=0, le=3, description="meals included")
    building_age_yr: float = Field(..., ge=0, description="age of the building")
    wifi_mbps: float = Field(..., ge=0, description="rule 2: below 0 is a typing mistake")
    furnishing: Literal["unfurnished", "semi", "full"]
    locality: Literal["Sector East", "Sector West", "Market Side"]


class RentResponse(BaseModel):
    prediction: float
    model_version: str


app = FastAPI(title="MP203 PG Room Rent (Kochi)", version="1.0")
model = joblib.load(MODEL_PATH)
logger.info("model loaded", extra={"fields": {"model_path": MODEL_PATH}})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=RentResponse)
def predict(request: RentRequest):
    start = time.perf_counter()
    row = pd.DataFrame([request.model_dump()])      # one-row table, same columns as training
    prediction = round(float(model.predict(row)[0]), 2)

    # The request holds only room details (no names or phone numbers),
    # so it is safe to log.
    logger.info("prediction", extra={"fields": {
        "event": "prediction",
        "inputs": request.model_dump(),
        "prediction": prediction,
        "model_version": MODEL_VERSION,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return RentResponse(prediction=prediction, model_version=MODEL_VERSION)

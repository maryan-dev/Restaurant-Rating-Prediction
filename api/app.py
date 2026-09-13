"""
api/app.py
----------
FastAPI backend for the Restaurant Rating Predictor.

Run from the project root:
    uvicorn api.app:app --reload --port 8000

On first run it will auto-train the model (see src/train_model.py) from
dataset/zomato.csv if models/ doesn't have saved artifacts yet.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "dataset"

sys.path.insert(0, str(ROOT_DIR / "src"))
import train_model as tm  # noqa: E402

OPTIONS_CANDIDATES = [
    DATASET_DIR / "options.json",
    ROOT_DIR / "options.json",
]

app = FastAPI(
    title="Zomato Restaurant Rating Predictor",
    description="Predicts a restaurant's Aggregate rating (0-5) from its attributes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_scaler = None
_encoders = None
_meta = None


def _load_artifacts():
    global _model, _scaler, _encoders, _meta
    if not (MODELS_DIR / "model.joblib").exists():
        print("No trained model found — training now from dataset/zomato.csv ...")
        tm.train()
    _model = joblib.load(MODELS_DIR / "model.joblib")
    _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    _encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    _meta = joblib.load(MODELS_DIR / "meta.joblib")


@app.on_event("startup")
def startup_event():
    _load_artifacts()


def _load_options() -> Optional[dict]:
    for path in OPTIONS_CANDIDATES:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _yes_no(value: Union[str, bool, int]) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return "Yes" if value else "No"
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "1"):
        return "Yes"
    if v in ("no", "n", "false", "0"):
        return "No"
    raise ValueError(f"Could not interpret '{value}' as Yes/No")


def _safe_encode(encoder, label: str, field_name: str) -> int:
    classes = list(encoder.classes_)
    if label not in classes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown value '{label}' for '{field_name}'. "
                f"See GET /options for the list of values the model was trained on."
            ),
        )
    return int(encoder.transform([label])[0])


class RestaurantFeatures(BaseModel):
    country_code: int = Field(..., alias="Country Code", description="Zomato country code, e.g. 1 = India")
    city: str = Field(..., alias="City")
    primary_cuisine: str = Field(..., alias="Primary Cuisine")
    num_cuisines: int = Field(1, alias="Num Cuisines", ge=1)
    average_cost_for_two_usd: float = Field(..., alias="Average Cost for two (USD)", ge=0)
    has_table_booking: Union[str, bool, int] = Field("No", alias="Has Table booking")
    has_online_delivery: Union[str, bool, int] = Field("No", alias="Has Online delivery")
    price_range: int = Field(..., alias="Price range", ge=1, le=4)
    votes: float = Field(0, alias="Votes", ge=0)
    votes_log: Optional[float] = Field(None, alias="Votes (log)", ge=0)
    city_restaurant_count: Optional[float] = Field(None, alias="City Restaurant Count", ge=0)

    model_config = {"populate_by_name": True}

    @field_validator("has_table_booking", "has_online_delivery")
    @classmethod
    def _normalize_yes_no(cls, v):
        return _yes_no(v)


class PredictionResponse(BaseModel):
    predicted_rating: float
    aggregate_rating: float
    rating_text: str
    interpretation: str


def _rating_text(rating: float) -> str:
    if rating >= 4.5:
        return "Excellent"
    if rating >= 4.0:
        return "Very Good"
    if rating >= 3.5:
        return "Good"
    if rating >= 2.5:
        return "Average"
    return "Poor"


@app.get("/")
def root():
    return {"status": "ok", "message": "Zomato rating predictor API. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/options")
def options():
    data = _load_options()
    if data is None:
        raise HTTPException(status_code=404, detail="options.json not found on the server.")
    return data


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: RestaurantFeatures):
    if _model is None:
        _load_artifacts()

    city_enc = _safe_encode(_encoders["City"], payload.city, "City")
    cuisine_enc = _safe_encode(_encoders["Primary Cuisine"], payload.primary_cuisine, "Primary Cuisine")
    booking_enc = _safe_encode(_encoders["Has Table booking"], payload.has_table_booking, "Has Table booking")
    delivery_enc = _safe_encode(_encoders["Has Online delivery"], payload.has_online_delivery, "Has Online delivery")

    votes = payload.votes
    votes_log = payload.votes_log if payload.votes_log is not None else float(np.log1p(votes))
    if payload.votes_log is not None and payload.votes == 0:
        # only votes_log was supplied by the client -> recover raw votes
        votes = float(np.expm1(payload.votes_log))

    city_count = payload.city_restaurant_count
    if city_count is None:
        city_count = _meta["city_counts"].get(payload.city, _meta["default_city_count"])

    cost = payload.average_cost_for_two_usd
    votes_capped = votes
    bounds = _meta["cap_bounds"]
    if "Votes" in bounds:
        votes_capped = min(max(votes, bounds["Votes"]["lower"]), bounds["Votes"]["upper"])
    if "Average Cost for two (USD)" in bounds:
        b = bounds["Average Cost for two (USD)"]
        cost = min(max(cost, b["lower"]), b["upper"])

    row = pd.DataFrame([{
        "Country Code": payload.country_code,
        "City_enc": city_enc,
        "Primary Cuisine_enc": cuisine_enc,
        "Num Cuisines": payload.num_cuisines,
        "Average Cost for two (USD)": cost,
        "Has Table booking_enc": booking_enc,
        "Has Online delivery_enc": delivery_enc,
        "Price range": payload.price_range,
        "Votes": votes_capped,
        "Votes (log)": votes_log,
        "City Restaurant Count": city_count,
    }])[tm.FEATURE_COLS]

    scaled = _scaler.transform(row)
    pred = float(_model.predict(scaled)[0])
    pred = max(0.0, min(5.0, pred))

    return PredictionResponse(
        predicted_rating=round(pred, 2),
        aggregate_rating=round(pred, 2),
        rating_text=_rating_text(pred),
        interpretation=_rating_text(pred),
    )

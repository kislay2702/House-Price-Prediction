"""
House Price Prediction - Flask Backend
========================================
Serves a web UI that collects ~20 key house features from the user,
fills in every remaining feature the trained model expects with a
statistically representative default (median / mode from the training
data), reproduces the exact feature-engineering used during training,
runs the fitted preprocessing pipeline, and returns a price prediction
from the trained Random Forest model.

Pipeline (must match training exactly):
    raw form fields + defaults
        -> engineered features (HouseAge, TotalSF, TotalBath, ...)
        -> preprocessor.pkl   (ColumnTransformer: OneHotEncoder + passthrough)
        -> scaler.pkl         (StandardScaler)
        -> rf.pkl             (RandomForestRegressor)
        -> predicted SalePrice

NOTE ON scaler.pkl:
    The original training notebook (model.ipynb) fit a StandardScaler on
    the training split and trained the Random Forest on the SCALED
    features, but never saved that scaler to disk. Feeding the model
    unscaled data (i.e. using preprocessor.pkl alone) produces unusable
    predictions (verified: R^2 of -10.1 on the held-out test set).
    We reconstructed the scaler by re-running the exact steps from
    model.ipynb (train_test_split(test_size=0.25, random_state=42) then
    StandardScaler().fit(x_train)) against the original training CSV.
    This does NOT retrain or modify the Random Forest in any way -
    it recovers a missing preprocessing artifact so the saved model
    receives the same kind of input it was trained on.
    With the scaler restored, test-set R^2 is 0.884.
"""

import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from schema import (
    CATEGORICAL_DEFAULTS,
    CATEGORY_OPTIONS,
    FEATURE_ORDER,
    NUMERIC_DEFAULTS,
)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once, at startup
# ---------------------------------------------------------------------------
try:
    with open(BASE_DIR / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "rf.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as exc:  # pragma: no cover - fails fast at boot
    raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc

# Fields exposed on the form. Everything else in FEATURE_ORDER is
# auto-filled from NUMERIC_DEFAULTS / CATEGORICAL_DEFAULTS.
FORM_NUMERIC_FIELDS = [
    "OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF",
    "GarageCars", "GarageArea", "YearBuilt", "YearRemodAdd",
    "LotArea", "FullBath", "HalfBath",
]
FORM_CATEGORICAL_FIELDS = [
    "Neighborhood", "KitchenQual", "ExterQual", "BsmtQual",
    "MSZoning", "HouseStyle", "GarageType", "CentralAir", "SaleCondition",
]

NUMERIC_BOUNDS = {
    "OverallQual": (1, 10), "OverallCond": (1, 10),
    "GrLivArea": (200, 10000), "TotalBsmtSF": (0, 6000),
    "GarageCars": (0, 5), "GarageArea": (0, 1500),
    "YearBuilt": (1870, 2026), "YearRemodAdd": (1870, 2026),
    "LotArea": (500, 250000), "FullBath": (0, 5), "HalfBath": (0, 3),
}


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame matching the trained schema exactly.

    Combines user-supplied values with data-driven defaults, then
    recreates every engineered feature the training notebook computed
    before the encoder was fit.
    """
    row = dict(NUMERIC_DEFAULTS)
    row.update(CATEGORICAL_DEFAULTS)

    # --- validate & apply user-supplied numeric fields -------------------
    for field in FORM_NUMERIC_FIELDS:
        raw = payload.get(field)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Missing value for '{field}'.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{field}' must be a number.")
        lo, hi = NUMERIC_BOUNDS[field]
        if not (lo <= value <= hi):
            raise ValueError(f"'{field}' must be between {lo} and {hi}.")
        row[field] = value

    # --- validate & apply user-supplied categorical fields ----------------
    for field in FORM_CATEGORICAL_FIELDS:
        value = payload.get(field)
        if not value:
            raise ValueError(f"Missing value for '{field}'.")
        valid = CATEGORY_OPTIONS[field]
        if value not in valid:
            raise ValueError(f"'{field}' must be one of {valid}.")
        row[field] = value

    if row["YearRemodAdd"] < row["YearBuilt"]:
        raise ValueError("Year Remodeled cannot be earlier than Year Built.")

    # -----------------------------------------------------------------
    # Derive raw fields the model needs that are NOT directly on the
    # form, from what the user *did* provide (better than a flat
    # median default). These heuristics are documented assumptions,
    # not values learned during training.
    # -----------------------------------------------------------------
    # Split GrLivArea into 1st/2nd floor SF based on the chosen house style.
    two_story_styles = {"2Story", "2.5Fin", "2.5Unf"}
    if row["HouseStyle"] in two_story_styles:
        row["1stFlrSF"] = round(row["GrLivArea"] * 0.55)
        row["2ndFlrSF"] = row["GrLivArea"] - row["1stFlrSF"]
    else:
        row["1stFlrSF"] = row["GrLivArea"]
        row["2ndFlrSF"] = 0

    # Split TotalBsmtSF into finished/unfinished using the training
    # data's average finished ratio (~40%).
    row["BsmtFinSF1"] = round(row["TotalBsmtSF"] * 0.40)
    row["BsmtUnfSF"] = row["TotalBsmtSF"] - row["BsmtFinSF1"]

    # Garage is usually built alongside the house; no garage -> year 0.
    row["GarageYrBlt"] = 0 if row["GarageType"] == "None" else row["YearBuilt"]

    # --- engineered features (reproduced exactly from eda.ipynb) ---------
    row["HouseAge"] = row["YrSold"] - row["YearBuilt"]
    row["RemodAge"] = row["YrSold"] - row["YearRemodAdd"]
    row["TotalBath"] = (
        row["FullBath"] + 0.5 * row["HalfBath"]
        + row["BsmtFullBath"] + 0.5 * row["BsmtHalfBath"]
    )
    row["TotalSF"] = row["TotalBsmtSF"] + row["1stFlrSF"] + row["2ndFlrSF"]
    row["TotalPorchSF"] = (
        row["WoodDeckSF"] + row["OpenPorchSF"] + row["EnclosedPorch"]
        + row["3SsnPorch"] + row["ScreenPorch"]
    )
    row["HasGarage"] = int(row["GarageArea"] > 0)
    row["HasBasement"] = int(row["TotalBsmtSF"] > 0)
    row["HasFireplace"] = int(row["Fireplaces"] > 0)
    row["HasPool"] = int(row["PoolArea"] > 0)
    row["TotalOutdoorSF"] = (
        row["PoolArea"] + row["WoodDeckSF"] + row["OpenPorchSF"]
        + row["EnclosedPorch"] + row["3SsnPorch"] + row["ScreenPorch"]
    )

    # MSSubClass must be a string category, exactly as in training
    # (df['MSSubClass'] = df['MSSubClass'].astype(str)).
    row["MSSubClass"] = str(row["MSSubClass"])

    return pd.DataFrame([row])[FEATURE_ORDER]


def predict_price(payload: dict) -> float:
    """Run the full preprocessing + model pipeline and return a price."""
    frame = build_feature_row(payload)
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = pd.DataFrame(transformed, columns=scaler.feature_names_in_)
    scaled = scaler.transform(transformed)
    prediction = model.predict(scaled)[0]
    return float(max(prediction, 0))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main form page."""
    return render_template(
        "index.html",
        neighborhoods=CATEGORY_OPTIONS["Neighborhood"],
        house_styles=CATEGORY_OPTIONS["HouseStyle"],
        garage_types=CATEGORY_OPTIONS["GarageType"],
        mszonings=CATEGORY_OPTIONS["MSZoning"],
        sale_conditions=CATEGORY_OPTIONS["SaleCondition"],
        quality_options=["Ex", "Gd", "TA", "Fa"],
        bsmt_quality_options=CATEGORY_OPTIONS["BsmtQual"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data (JSON), return a predicted price as JSON."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        price = predict_price(payload)
        return jsonify({
            "success": True,
            "predicted_price": round(price, 2),
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception:
        app.logger.error("Prediction failed:\n%s", traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Something went wrong while generating the prediction.",
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(debug=True)

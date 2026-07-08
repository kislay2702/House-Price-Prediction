"# House-Price-Prediction-Model-" 


A production-styled Flask web app that serves a Random Forest model trained
on the Ames Housing dataset. Enter ~20 key property details and get an
instant price estimate through a glassmorphism, luxury-real-estate-themed UI.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Flask](https://img.shields.io/badge/flask-3.x-black)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-orange)

## ✨ Features

- **Random Forest regression** on 282 engineered/encoded features, trained on the Ames Housing dataset (R² ≈ 0.88 on held-out test data)
- **20-field form** covering the highest-impact features (quality, size, garage, location, age, sale condition) — every other feature the model expects is auto-filled from the training data's median/mode
- Modern glassmorphism UI: hero section, gradient buttons, animated price reveal, session-only prediction history
- Server- and client-side validation with friendly error messaging
- Modular, PEP8-compliant Flask backend with exception handling

## 🧠 About the model & pipeline

This app loads three trained artifacts:

| File | What it is |
|---|---|
| `preprocessor.pkl` | `ColumnTransformer` — one-hot encodes 44 categorical columns (`drop='first'`), passes 45 numeric/engineered columns through |
| `scaler.pkl` | `StandardScaler` fit on the same training split used for the model |
| `rf.pkl` | `RandomForestRegressor` (100 trees), trained on the scaled, encoded features |

**Why `scaler.pkl` exists.** The original training notebook fit a
`StandardScaler` on the training split and trained the Random Forest on the
*scaled* output — but the scaler itself was never saved, only the
`ColumnTransformer` and the model were. Feeding the model unscaled features
produces essentially random output (verified: test-set R² of **-10.1**,
worse than predicting the mean every time). We reconstructed the exact
missing scaler by re-running the documented steps from the training
notebook (`train_test_split(test_size=0.25, random_state=42)` →
`StandardScaler().fit(x_train)`) against the original training CSV. This
does not retrain or alter the Random Forest in any way — it recovers a
missing preprocessing step so the model receives the same shape of input
it was trained on. With the scaler restored, test-set R² is **0.884** and
MAE is **≈ $16,858**.

### Request flow

```
Form input (20 fields)
  -> merged with median/mode defaults for the remaining ~69 raw features
  -> engineered features recreated (HouseAge, TotalSF, TotalBath, HasGarage, ...)
  -> preprocessor.pkl  (one-hot encode + passthrough)
  -> scaler.pkl        (standardize)
  -> rf.pkl             (predict)
  -> price returned as JSON
```

## 📁 Project structure

```
House_Price_Prediction/
│
├── app.py                 # Flask app: routes, validation, prediction pipeline
├── schema.py               # Auto-derived feature order + medians/modes
├── rf.pkl                  # Trained RandomForestRegressor (not modified)
├── preprocessor.pkl        # Trained ColumnTransformer (not modified)
├── scaler.pkl               # Reconstructed StandardScaler (see above)
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    ├── script.js
    └── images/
```

## 🚀 Running locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Visit `http://127.0.0.1:5000`.

## ☁️ Deploying on Render

1. Push this folder to a GitHub repository.
2. On Render, create a new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy — Render will assign a public URL.

## 🧾 Form fields → model features

The form collects: Overall Quality, Overall Condition, Ground Living Area,
Total Basement Area, Garage Cars, Garage Area, Garage Type, Year Built,
Year Remodeled, Lot Area, Full Bathrooms, Half Bathrooms, Neighborhood,
Kitchen Quality, Exterior Quality, Basement Quality, MS Zoning, House
Style, Central Air, and Sale Condition.

Everything else (porch square footage, basement finish split, garage
year, misc. features, etc.) is filled in automatically — either from the
training data's median/mode, or derived from what you entered (e.g. 1st/2nd
floor split from Ground Living Area + House Style). See the comments in
`app.py::build_feature_row` for the exact logic.

## ⚠️ Disclaimer

This tool produces a statistical estimate based on a model trained on a
specific historical dataset (Ames, Iowa property sales). It is a portfolio
demonstration, not a substitute for a professional real-estate appraisal.

## 🛠️ Tech stack

Flask · pandas · NumPy · scikit-learn · vanilla JavaScript · CSS (glassmorphism)

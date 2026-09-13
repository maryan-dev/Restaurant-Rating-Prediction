"""
src/train_model.py
-------------------
Reproduces the cleaning / feature-engineering steps from
`notebooks/zomato_pipeline.ipynb`, then trains and compares FOUR regression
algorithms on the *same* train/test split so the comparison is fair:

    1. Linear Regression      (bootcamp)
    2. Decision Tree          (bootcamp)
    3. Random Forest          (bootcamp)
    4. XGBoost                (researched independently)

Selection rule (stated up front, per assignment requirements):
    The best model is the one with the LOWEST RMSE on the held-out test set.
    RMSE is used (over plain MAE) because it penalizes larger errors more
    heavily, which matters for a rating scale where a 2-star miss is far
    worse than a 0.2-star miss.

Only the winning model is refit on the full dataset and saved to models/
for the API to load. All four models' metrics are also saved there
(metrics.json) for the README / project paper comparison table.

Run from the project root:
    python src/train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"

FEATURE_COLS = [
    "Country Code",
    "City_enc",
    "Primary Cuisine_enc",
    "Num Cuisines",
    "Average Cost for two (USD)",
    "Has Table booking_enc",
    "Has Online delivery_enc",
    "Price range",
    "Votes",
    "Votes (log)",
    "City Restaurant Count",
]
TARGET_COL = "Aggregate rating"

FX_TO_USD = {
    "Indian Rupees(Rs.)": 0.012, "Dollar($)": 1.0, "Pounds(\u00a3)": 1.27,
    "Brazilian Real(R$)": 0.19, "Emirati Diram(AED)": 0.27, "Rand(R)": 0.055,
    "NewZealand($)": 0.61, "Turkish Lira(TL)": 0.031, "Botswana Pula(P)": 0.074,
    "Indonesian Rupiah(IDR)": 0.000064, "Qatari Rial(QR)": 0.27, "Sri Lankan Rupee(LKR)": 0.0031,
}

SELECTION_RULE = "lowest RMSE on the held-out test set"


def find_zomato_csv() -> Path:
    candidates = [
        DATASET_DIR / "zomato.csv",
        ROOT_DIR / "zomato.csv",
        SRC_DIR / "zomato.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find zomato.csv. Expected it at dataset/zomato.csv "
        "relative to the project root."
    )


def iqr_cap(series: pd.Series) -> tuple[pd.Series, float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower, upper), lower, upper


def build_dataset() -> tuple[pd.DataFrame, dict, dict]:
    csv_path = find_zomato_csv()
    df = pd.read_csv(csv_path, encoding="latin-1")

    df = df.drop(columns=["Longitude", "Latitude"])

    # currency -> USD
    df["Currency"] = df["Currency"].str.replace("\x8c", "", regex=False).str.strip()
    df["fx_to_usd"] = df["Currency"].map(FX_TO_USD)
    df["Average Cost for two (USD)"] = (df["Average Cost for two"] * df["fx_to_usd"]).round(2)

    # normalize string / binary columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    binary_cols = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
    for c in binary_cols:
        df[c] = df[c].str.capitalize()

    if df["Switch to order menu"].nunique() == 1:
        df = df.drop(columns=["Switch to order menu"])

    # cuisines
    df["Cuisines"] = df["Cuisines"].replace("nan", np.nan)
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # de-dup
    df = df.drop_duplicates(subset="Restaurant ID", keep="first")
    df = df.drop_duplicates()

    # outlier capping (bounds saved for use at inference time)
    cap_bounds = {}
    for col in ["Votes", "Average Cost for two (USD)"]:
        df[col], lower, upper = iqr_cap(df[col])
        cap_bounds[col] = {"lower": float(lower), "upper": float(upper)}

    # label encoders (only the ones used downstream as model features)
    encoders = {}
    for c in ["City", "Has Table booking", "Has Online delivery"]:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c])
        encoders[c] = le

    # derived features
    df["Primary Cuisine"] = df["Cuisines"].apply(lambda x: x.split(",")[0].strip())
    df["Num Cuisines"] = df["Cuisines"].apply(lambda x: len(x.split(",")))
    df["Is Rated"] = (df["Aggregate rating"] > 0).astype(int)
    df["Votes (log)"] = np.log1p(df["Votes"])

    city_counts = df["City"].value_counts()
    df["City Restaurant Count"] = df["City"].map(city_counts)

    le_prim = LabelEncoder()
    df["Primary Cuisine_enc"] = le_prim.fit_transform(df["Primary Cuisine"])
    encoders["Primary Cuisine"] = le_prim

    meta = {
        "cap_bounds": cap_bounds,
        "city_counts": city_counts.to_dict(),
        "default_city_count": int(city_counts.median()),
    }

    return df, encoders, meta


def get_candidate_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        ),
    }


def evaluate(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    return {"Algorithm": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def sanity_checks(model, scaler, X_test_raw, y_test, n=3):
    print(f"\nSanity checks on {n} sample predictions (winning model):")
    sample = X_test_raw.sample(n=n, random_state=1)
    scaled = scaler.transform(sample)
    preds = model.predict(scaled)
    for (idx, row), pred in zip(sample.iterrows(), preds):
        actual = y_test.loc[idx]
        print(f"  input={row.to_dict()}")
        print(f"    -> predicted={pred:.2f}  actual={actual:.2f}\n")


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, encoders, meta = build_dataset()
    model_df = df[df["Is Rated"] == 1].copy()

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]

    # SAME split used for every algorithm -> fair comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

    results = []
    fitted_models = {}
    for name, model in get_candidate_models().items():
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model
        results.append(evaluate(name, model, X_test_scaled, y_test))

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== Model comparison (same train/test split, test_size=0.2, random_state=42) ===")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["Algorithm"]
    print(f"\nSelection rule: {SELECTION_RULE}")
    print(f"Winner: {best_name}\n")

    sanity_checks(fitted_models[best_name], scaler, X_test, y_test, n=3)

    # refit scaler + winning model on ALL rated rows for the deployed artifact
    final_scaler = StandardScaler()
    X_all_scaled = pd.DataFrame(final_scaler.fit_transform(X), columns=FEATURE_COLS, index=X.index)
    final_model = get_candidate_models()[best_name]
    final_model.fit(X_all_scaled, y)

    joblib.dump(final_model, MODELS_DIR / "model.joblib")
    joblib.dump(final_scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump(meta, MODELS_DIR / "meta.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "selection_rule": SELECTION_RULE,
            "winner": best_name,
            "comparison": results,
            "n_train_rows": len(X),
        }, f, indent=2)

    print(f"Saved winning model ('{best_name}') and artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    train()

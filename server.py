
import os, sqlite3, joblib, numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Safe model loader + fallback (paste near top, after imports) ---
import os
import joblib
import traceback
import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(__file__), "eff_model.joblib")
HEALTH_FILE = os.path.join(os.path.dirname(__file__), "health_model.joblib")

eff_model = None
health_model = None

def local_rule_efficiency(arr):
    # arr is [rpm, throttle, temperature, load, afr, traffic, distance]
    rpm, throttle, temp, load, afr, traffic, distance = arr
    score = 100.0
    if rpm < 1500:
        score -= 15
    elif rpm > 3000:
        score -= 20
    if throttle > 70:
        score -= 10
    if afr < 13:
        score -= 18
    elif afr > 16:
        score -= 12
    if temp < 70:
        score -= 10
    elif temp > 100:
        score -= 8
    if load > 100:
        score -= 15
    score -= traffic * 0.1
    if distance < 3:
        score -= 10
    elif distance > 30:
        score += 5
    return float(max(5, min(100, score)))

# Try loading joblib models, but fallback gracefully if anything goes wrong
try:
    if os.path.exists(MODEL_FILE):
        eff_model = joblib.load(MODEL_FILE)
        print(f"Loaded eff_model from {MODEL_FILE}")
    else:
        print(f"eff_model.joblib not found at {MODEL_FILE} — using local rule fallback")

    if os.path.exists(HEALTH_FILE):
        health_model = joblib.load(HEALTH_FILE)
        print(f"Loaded health_model from {HEALTH_FILE}")
    else:
        print(f"health_model.joblib not found at {HEALTH_FILE} — health checks will use simple rules")
except Exception as e:
    print("Error loading models (continuing with safe fallback). Traceback:")
    traceback.print_exc()
    eff_model = None
    health_model = None

# Helper to call eff_model.predict or fallback
def predict_efficiency_from_model(arr):
    """
    arr: list-like [rpm, throttle, temperature, load, afr, traffic, distance]
    returns float ml_eff
    """
    global eff_model
    try:
        if eff_model is not None:
            # ensure correct shape
            X = np.array(arr, dtype=float).reshape(1, -1)
            pred = eff_model.predict(X)
            return float(pred[0])
        else:
            return local_rule_efficiency(arr)
    except Exception:
        # if any model error occurs, fallback to rule based and print trace
        print("Model prediction failed; falling back to rule-based. Traceback:")
        traceback.print_exc()
        return local_rule_efficiency(arr)

# Helper health check if health_model present
def predict_health_indicator(a):
    """
    a: array-like [afr, temperature, load]
    returns (status_text, score)
    """
    try:
        if health_model is not None:
            pred = health_model.predict(np.array(a, dtype=float).reshape(1, -1))[0]
            if pred == -1:
                return ("⚠ Check engine variables", 40)
            else:
                return ("✔ Engine normal", 90)
        else:
            # simple rule-based health
            afr, temp, loadv = a
            if afr < 12 or afr > 17 or temp > 105 or loadv > 150:
                return ("⚠ Check engine variables", 40)
            return ("✔ Engine normal", 90)
    except Exception:
        traceback.print_exc()
        return ("⚠ Check engine variables", 40)
# --- end safe model loader ---



MODEL_FILE = "eff_model.joblib"
HEALTH_FILE = "health_model.joblib"
DB_FILE = "fuel_ai_service.db"
STATIC_DIR = "static"

app = FastAPI(title="FERA")
app.add_middleware(CORSMiddleware, allow_origins=["*"],  allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Telemetry(BaseModel):
    rpm: int
    throttle: float
    temperature: float
    load: float
    afr: float
    traffic: float
    distance: float
    fuel_type: str = "petrol"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rpm INTEGER, throttle REAL, temperature REAL, load REAL,
            afr REAL, traffic REAL, distance REAL, fuel_type TEXT,
            ml_eff REAL, rule_eff REAL, trip_cost REAL, fuel_unit TEXT,
            health_score REAL, health_status TEXT, recommendation TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_entry(d, ml_eff, r_eff, cost, unit, hscore, hstatus, recs):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO logs (rpm, throttle, temperature, load, afr, traffic, distance, fuel_type,
            ml_eff, rule_eff, trip_cost, fuel_unit, health_score, health_status, recommendation, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        d["rpm"], d["throttle"], d["temperature"], d["load"], d["afr"],
        d["traffic"], d["distance"], d.get("fuel_type","petrol"), ml_eff, r_eff, cost, unit,
        hscore, hstatus, "; ".join(recs), datetime.now(timezone.utc).isoformat()
    ))
    conn.commit()
    conn.close()

def train_models_quick(X_df, y_series):
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    X = X_df[["rpm","throttle","temperature","load","afr","traffic","distance"]]
    y = y_series
    eff = RandomForestRegressor(n_estimators=100, random_state=42)
    eff.fit(X, y)
    health = IsolationForest(contamination=0.05, random_state=42)
    health.fit(X_df[["afr","temperature","load"]])
    joblib.dump(eff, MODEL_FILE)
    joblib.dump(health, HEALTH_FILE)
    return eff, health

def load_models():
    # Try load; if it fails, train on synthetic data and save models (fallback)
    try:
        eff = joblib.load(MODEL_FILE)
        health = joblib.load(HEALTH_FILE)
        return eff, health
    except Exception as e:
        print("Model load failed:", e)
        print("Training fallback models (this will take ~30-90s) ...")
        # create synthetic dataset similar to frontend heuristics
        import numpy as np, pandas as pd
        N = 3000
        rng = np.random.default_rng(42)
        data = {
            "rpm": rng.integers(800, 6000, N),
            "throttle": rng.integers(0, 100, N),
            "temperature": rng.uniform(50, 110, N),
            "load": rng.integers(10, 200, N),
            "afr": rng.uniform(10, 18, N),
            "traffic": rng.integers(0, 100, N),
            "distance": rng.uniform(1, 100, N),
        }
        df = pd.DataFrame(data)
        def true_efficiency(row):
            score = 100.0
            if row.rpm < 1500: score -= 15
            elif row.rpm > 3000: score -= 20
            if row.throttle > 70: score -= 10
            if row.afr < 13: score -= 18
            elif row.afr > 16: score -= 12
            if row.temperature < 70: score -= 10
            elif row.temperature > 100: score -= 8
            if row.load > 100: score -= 15
            score -= row.traffic * 0.1
            if row.distance < 3: score -= 10
            elif row.distance > 30: score += 5
            return max(5, min(100, score))
        df["efficiency"] = df.apply(true_efficiency, axis=1)
        return train_models_quick(df, df["efficiency"])

from fastapi.responses import JSONResponse
import traceback

@app.post("/predict")
async def predict(payload: dict):
    try:
        # ---------------------------
        # Extract values safely
        # ---------------------------
        d = {
            "rpm": float(payload.get("rpm", 0)),
            "throttle": float(payload.get("throttle", 0)),
            "temperature": float(payload.get("temperature", 0)),
            "load": float(payload.get("load", 0)),
            "afr": float(payload.get("afr", 0)),
            "traffic": float(payload.get("traffic", 0)),
            "distance": float(payload.get("distance", 0)),
            "fuel_type": payload.get("fuel_type", "petrol")
        }

        # ---------------------------
        # Run your model predictions
        # ---------------------------
        ml_eff = float(eff_model.predict([[ 
            d["rpm"], d["throttle"], d["temperature"], d["load"],
            d["afr"], d["traffic"], d["distance"]
        ]])[0])

        # Compute kmpl like before
        if d["fuel_type"] == "diesel":
            kmpl = max(12, min(24, ml_eff / 5.0))
            fuel_unit = "L"
        elif d["fuel_type"] == "cng":
            kmpl = max(10, min(28, ml_eff / 4.5))
            fuel_unit = "kg"
        else:
            kmpl = max(10, min(20, ml_eff / 6.0))
            fuel_unit = "L"

        fuel_used = d["distance"] / kmpl

        return {
            "ml_eff": ml_eff,
            "kmpl": kmpl,
            "fuel_used": fuel_used,
            "fuel_unit": fuel_unit
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("\n\n=== PREDICT ERROR TRACEBACK ===\n", tb, "\n===============================\n")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": tb}
        )

@app.get("/history")
def history(limit: int = 30):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(f"SELECT id, rpm, throttle, temperature, load, afr, traffic, distance, fuel_type, ml_eff, rule_eff, trip_cost, fuel_unit, health_score, health_status, recommendation, timestamp FROM logs ORDER BY id DESC LIMIT {limit}").fetchall()
    conn.close()
    cols = ["id","rpm","throttle","temperature","load","afr","traffic","distance","fuel_type","ml_eff","rule_eff","trip_cost","fuel_unit","health_score","health_status","recommendation","timestamp"]
    results = [dict(zip(cols, r)) for r in rows]
    return results[::-1]

if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=9000, log_level="info")





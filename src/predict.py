from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"

def predict_delivery(distance_km, traffic_level, weather_bad, package_weight_kg):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    sample = pd.DataFrame([{
        "distance_km": distance_km,
        "traffic_level": traffic_level,
        "weather_bad": weather_bad,
        "package_weight_kg": package_weight_kg,
    }], columns=features)
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    return prediction, probability

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python src/train.py` first.")
    result, probability = predict_delivery(95, 2, 1, 8)
    print("Prediction:", "Delayed" if result else "On time")
    print(f"Delay probability: {probability:.1%}")

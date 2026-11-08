from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"

def test_model_predicts_binary_class():
    assert MODEL_PATH.exists(), "Run `python src/train.py` first."
    saved = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "distance_km": 50,
        "traffic_level": 1,
        "weather_bad": 0,
        "package_weight_kg": 4,
    }], columns=saved["features"])
    assert int(saved["model"].predict(sample)[0]) in (0, 1)

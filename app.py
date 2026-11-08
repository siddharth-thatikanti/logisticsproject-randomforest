from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"

st.set_page_config(page_title="Smart Logistics", page_icon="🚚", layout="centered")

st.title("🚚 Smart Logistics")
st.subheader("Delivery Delay Prediction")
st.write("Enter delivery details and let the Random Forest model estimate the risk of delay.")

if not MODEL_PATH.exists():
    st.warning("The trained model has not been created yet.")
    st.code("python src/train.py", language="bash")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

with st.form("delivery_form"):
    st.markdown("### Delivery details")
    distance = st.number_input("Distance to destination (km)", min_value=1.0, max_value=5000.0, value=95.0, step=1.0)
    traffic_label = st.selectbox("Traffic condition", ["Low", "Medium", "High"])
    weather_label = st.selectbox("Bad weather?", ["No", "Yes"])
    weight = st.number_input("Package weight (kg)", min_value=0.1, max_value=1000.0, value=8.0, step=0.5)
    submitted = st.form_submit_button("Predict delivery status", type="primary", use_container_width=True)

if submitted:
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    weather_bad = 1 if weather_label == "Yes" else 0
    sample = pd.DataFrame([{
        "distance_km": distance,
        "traffic_level": traffic_map[traffic_label],
        "weather_bad": weather_bad,
        "package_weight_kg": weight,
    }], columns=features)

    prediction = int(model.predict(sample)[0])
    delay_probability = float(model.predict_proba(sample)[0][1])

    st.markdown("---")
    st.markdown("### Prediction")
    if prediction == 1:
        st.error("⚠️ The model predicts that this delivery may be delayed.")
    else:
        st.success("✅ The model predicts that this delivery will arrive on time.")

    st.metric("Estimated delay probability", f"{delay_probability:.1%}")
    st.progress(delay_probability)
    st.caption("This is a demonstration model trained on a tiny sample dataset. Do not use its prediction for real operational decisions.")

with st.expander("About this model"):
    st.write("Algorithm: RandomForestClassifier")
    st.write("Features: distance, traffic, weather, and package weight.")
    st.write("Target: delivery delayed (0 = on time, 1 = delayed).")

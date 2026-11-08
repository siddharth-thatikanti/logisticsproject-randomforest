# Smart Logistics – Random Forest with GUI

A small logistics delivery-delay prediction demo using Random Forest and a Streamlit web interface.

## Features
- Train a Random Forest classifier from a sample CSV dataset
- Evaluate accuracy, classification report, and confusion matrix
- Predict whether a delivery is likely to be delayed
- Interactive GUI built with Streamlit

## Input fields
- Distance (km)
- Traffic level (Low / Medium / High)
- Bad weather (No / Yes)
- Package weight (kg)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train and save the model
python src/train.py

# Start the GUI
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

If the model file does not exist, train it first with `python src/train.py`.

## Dataset note
The included CSV is a tiny illustrative sample for learning only. Its scores are not evidence of real-world performance. Use a larger, representative logistics dataset and time-based validation before any real deployment.

No Dockerfile is included.

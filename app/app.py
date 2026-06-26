import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = os.path.join(BASE_DIR, "models", "Decision_Tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "processed_data", "columns.joblib")

# Categorical encoding maps — must match main.py's .astype('category').cat.codes
# Codes are assigned in alphabetical order by pandas when using cat.codes.
CATEGORY_CODES = {
    "Department": {
        "customer support": 0,
        "engineering": 1,
        "finance": 2,
        "hr": 3,
        "it": 4,
        "legal": 5,
        "marketing": 6,
        "operations": 7,
        "sales": 8,
    },
    "Gender": {
        "female": 0,
        "male": 1,
        "other": 2,
    },
    "Education_Level": {
        "bachelor": 0,
        "high school": 1,
        "master": 2,
        "phd": 3,
    },
    # Resigned: pandas encodes bool False < True → 0, 1
    "Resigned": {
        "false": 0,
        "no": 0,
        "0": 0,
        "true": 1,
        "yes": 1,
        "1": 1,
    },
}

model, scaler, columns = None, None, None

try:
    print(f"📂 Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

    print(f"📂 Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully!")

    print(f"📂 Loading columns from: {COLUMNS_PATH}")
    columns = joblib.load(COLUMNS_PATH)
    print("✅ Columns loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model/scaler/columns: {e}")


def encode_categorical(key, val):
    """Return the integer category code for a categorical field.

    Matching is case-insensitive. Returns None if the value is not recognised.
    """
    mapping = CATEGORY_CODES.get(key)
    if mapping is None:
        return None
    return mapping.get(str(val).strip().lower())


def build_input_series(data):
    """Convert a raw JSON payload dict into a numeric Series aligned to *columns*.

    Defaults are the scaler means so that missing features don't skew the
    prediction — the same strategy used during training where NaNs were filled
    with column means before scaling.
    """
    # Start from scaler means as sensible defaults for missing fields
    if hasattr(scaler, 'mean_') and len(scaler.mean_) == len(columns):
        input_series = pd.Series(scaler.mean_.copy(), index=columns, dtype=float)
    else:
        input_series = pd.Series(0.0, index=columns, dtype=float)

    for key, val in data.items():
        # Handle categorical columns via the encoding map
        code = encode_categorical(key, val)
        if code is not None:
            if key in input_series.index:
                input_series[key] = float(code)
            continue

        # Handle boolean Resigned passed as a Python bool (not a string)
        if key == "Resigned" and isinstance(val, bool):
            if "Resigned" in input_series.index:
                input_series["Resigned"] = 1.0 if val else 0.0
            continue

        # Numeric fields — attempt direct float conversion
        if key in input_series.index:
            try:
                input_series[key] = float(val)
            except (TypeError, ValueError):
                pass  # leave the default in place

    # Derive the Efficiency feature the same way main.py does, but only when
    # the caller did not supply it explicitly and the required inputs are present.
    if "Efficiency" in input_series.index and "Efficiency" not in data:
        if "Monthly_Salary" in data and "Work_Hours_Per_Week" in data:
            try:
                salary = float(data["Monthly_Salary"])
                hours = float(data["Work_Hours_Per_Week"])
                input_series["Efficiency"] = salary / (hours + 1)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return input_series


@app.route('/', methods=['GET'])
def health():
    """Simple health-check / welcome endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Corporate Resource Optimization API is running.",
        "model_loaded": model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or columns is None:
        return jsonify({
            "status": "error",
            "message": "Model, scaler, or columns not loaded. "
                       "Run main.py to train and save the model first.",
        }), 503

    data = request.get_json(silent=True)
    print("📩 Received data:", data)

    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Invalid or empty JSON payload. "
                       "Send a JSON object with employee feature fields.",
        }), 400

    try:
        input_series = build_input_series(data)
        input_df = pd.DataFrame([input_series])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


if __name__ == '__main__':
    print("🚀 Flask API starting on 0.0.0.0:5002 ...")
    app.run(host='0.0.0.0', port=5002, debug=False)


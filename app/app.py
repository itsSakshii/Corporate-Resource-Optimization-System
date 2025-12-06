import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


MODEL_PATH = os.path.join(BASE_DIR, "models", "Decision_Tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "processed_data", "columns.joblib")


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


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, columns
    if model is None or scaler is None or columns is None:
        return jsonify({
            "status": "error",
            "message": "Model or scaler not loaded properly."
        })

    try:
        data = request.get_json()
        print("📩 Received data:", data)

        # Validate incoming JSON payload
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid or empty JSON payload"
            }), 400

       
        import pandas as pd

        
        if hasattr(scaler, 'mean_') and len(getattr(scaler, 'mean_')) == len(columns):
            input_series = pd.Series(getattr(scaler, 'mean_').copy(), index=columns, dtype=float)
        else:
            input_series = pd.Series(0, index=columns, dtype=float)

        
        def set_one_hot(key, val):
            key_prefix = f"{key}_"
            matched = False
            for col in columns:
                if col.startswith(key_prefix):
                    suffix = col[len(key_prefix):]
                    if str(suffix).strip().lower() == str(val).strip().lower():
                        input_series[col] = 1.0
                        matched = True
                    else:
                       
                        input_series[col] = 0.0
            return matched

        for key, val in data.items():
            
            if key in input_series.index:
                try:
                    input_series[key] = float(val)
                except Exception:
                    
                    set_one_hot(key, val)
            else:
                
                set_one_hot(key, val)

        
        if 'Resigned' in data and 'Resigned_True' in input_series.index:
            v = data.get('Resigned')
            if isinstance(v, bool):
                input_series['Resigned_True'] = 1.0 if v else 0.0
            else:
                if str(v).strip().lower() in ('yes', 'true', '1'):
                    input_series['Resigned_True'] = 1.0
                else:
                    input_series['Resigned_True'] = 0.0

        
        input_df = pd.DataFrame([input_series])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })


if __name__ == '__main__':
    print("🚀 Flask API starting...")
    app.run(host='127.0.0.1', port=5002, debug=True)

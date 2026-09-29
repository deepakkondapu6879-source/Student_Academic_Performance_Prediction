"""Backend API + static frontend. Run: python app.py  ->  http://127.0.0.1:5000"""
import joblib, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
app = Flask(__name__, static_folder="frontend", static_url_path="")
try: B = joblib.load("models/model.joblib")
except FileNotFoundError: B = None

def need_model():
    return None if B else (jsonify(error="No trained model. Run: python train.py"), 503)

@app.get("/")
def index(): return send_from_directory("frontend", "index.html")

@app.get("/api/schema")
def schema():
    return need_model() or jsonify(model=B["name"], features=B["features"], ranges=B["ranges"])

@app.post("/api/predict")
def predict():
    if not B: return need_model()
    d = request.get_json(silent=True) or {}
    miss = [f for f in B["features"] if f not in d]
    if miss: return jsonify(error="Missing features", missing=miss), 400
    try: x = pd.DataFrame([{f: float(d[f]) for f in B["features"]}])
    except (TypeError, ValueError): return jsonify(error="All feature values must be numbers"), 400
    p = float(B["model"].predict_proba(x)[0, 1])
    return jsonify(risk=round(p, 4), status="High" if p >= .6 else "Moderate" if p >= .35 else "Low")

if __name__ == "__main__": app.run(debug=False)

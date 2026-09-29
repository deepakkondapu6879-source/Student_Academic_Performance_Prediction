# Student Academic Performance Prediction
K. Deepak Kumar (44611089) · K.L.V. Subba Reddy (44611349) · K. Jagadish Sai (446011090) · M. Rohan (44611097)

## Contents
- `train.py` - trains Logistic Regression, SVM, Random Forest, Gradient Boosting (SMOTE + 5-fold CV grid search), saves best model, `models/results.csv` (paper Table II) and `models/shap_summary.png`
- `app.py` - Flask backend: `GET /api/schema`, `POST /api/predict`, serves the frontend
- `frontend/index.html` - dashboard (also works standalone: trains a logistic model in the browser from any CSV)
- `generate_sample_data.py` - SYNTHETIC demo data (replace with your real student CSV)
- `docs/` - IEEE paper (fill highlighted results from `models/results.csv`)

## Run
    pip install -r requirements.txt
    python generate_sample_data.py     # or put your CSV at data/students.csv
    python train.py                    # prints and saves real metrics
    python app.py                      # http://127.0.0.1:5000

CSV format: numeric feature columns + a 0/1 target named `at_risk` (or last column); text columns such as name are ignored.

## API example
    curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
      -d '{"attendance":62,"internal_marks":45,"study_hours":6,"assignments_done":4,"previous_gpa":5.8,"backlogs":2}'

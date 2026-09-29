"""Train + compare 4 classifiers. Usage: python train.py [data/students.csv]
Outputs: models/model.joblib, models/results.csv (paper Table II), models/shap_summary.png"""
import sys, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

path = sys.argv[1] if len(sys.argv) > 1 else "data/students.csv"
df = pd.read_csv(path)
target = next((c for c in df.columns if c.lower() in ("at_risk", "target", "label")), df.columns[-1])
y = df[target].astype(int)
X = df.drop(columns=[target]).select_dtypes("number")          # drops name/ID text columns
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
grids = {
 "Logistic Regression": (LogisticRegression(max_iter=2000), {"m__C": [.1, 1, 10]}),
 "SVM (RBF)": (SVC(probability=True, random_state=42), {"m__C": [1, 10], "m__gamma": ["scale", .1]}),
 "Random Forest": (RandomForestClassifier(random_state=42), {"m__n_estimators": [200, 400], "m__max_depth": [None, 8]}),
 "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {"m__n_estimators": [100, 200], "m__learning_rate": [.05, .1]})}
cv = StratifiedKFold(5, shuffle=True, random_state=42); rows, best = [], (None, -1, None)
for name, (est, g) in grids.items():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("smote", SMOTE(random_state=42)), ("m", est)])   # SMOTE only fits on training folds
    gs = GridSearchCV(pipe, g, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
    p = gs.predict(Xte)
    r = dict(Model=name, CV_F1=round(gs.best_score_*100, 1), Accuracy=round(accuracy_score(yte, p)*100, 1),
             Precision=round(precision_score(yte, p)*100, 1), Recall=round(recall_score(yte, p)*100, 1), F1=round(f1_score(yte, p)*100, 1))
    rows.append(r); print(r); print(confusion_matrix(yte, p))
    if r["F1"] > best[1]: best = (name, r["F1"], gs.best_estimator_)
res = pd.DataFrame(rows); res.to_csv("models/results.csv", index=False); print("\n", res.to_string(index=False))
joblib.dump({"model": best[2], "features": list(X.columns), "name": best[0],
             "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X}}, "models/model.joblib")
print("Best model:", best[0])
try:                                                              # explainability (paper Sec. IV-F)
    import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = lambda a: best[2].predict_proba(pd.DataFrame(a, columns=X.columns))[:, 1]
    ex = shap.Explainer(f, Xtr.sample(min(100, len(Xtr)), random_state=1))(Xte)
    shap.summary_plot(ex, Xte, show=False); plt.tight_layout(); plt.savefig("models/shap_summary.png", dpi=200)
except Exception as e: print("SHAP skipped:", e)

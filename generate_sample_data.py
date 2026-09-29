"""Creates data/students.csv with SYNTHETIC students. Replace with your real data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42); n = 500
d = pd.DataFrame({
 "student": [f"Student {i+1}" for i in range(n)],
 "attendance": rng.normal(75, 15, n).clip(30, 100).round(1),
 "internal_marks": rng.normal(60, 18, n).clip(0, 100).round(1),
 "study_hours": rng.normal(12, 5, n).clip(0, 30).round(1),
 "assignments_done": rng.normal(7, 2.5, n).clip(0, 10).round(),
 "previous_gpa": rng.normal(6.8, 1.4, n).clip(2, 10).round(2),
 "backlogs": rng.normal(1, 1.3, n).clip(0, 6).round()})
z = (-.09*(d.attendance-75) - .07*(d.internal_marks-60) - .12*(d.study_hours-12)
     - .25*(d.assignments_done-7) - .9*(d.previous_gpa-6.8) + .6*(d.backlogs-1) - .6
     + rng.normal(0, 1, n))
d["at_risk"] = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)
d.to_csv("data/students.csv", index=False); print(d.at_risk.mean(), "at-risk rate; saved data/students.csv")

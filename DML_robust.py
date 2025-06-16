import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import LinearDML

# Load data
df = pd.read_csv("cps_data_final_encoded.csv")
Y = df['log_wage'].values
D = df['male'].values
X = df.drop(columns=['log_wage', 'male']).values

# Model combo 1: Lasso + Logistic
model_y1 = LassoCV(cv=5)
model_t1 = LogisticRegressionCV(cv=5, solver='liblinear')
dml1 = LinearDML(model_y=model_y1, model_t=model_t1, discrete_treatment=True, cv=3, random_state=42)
dml1.fit(Y, D, X=X)
ate1 = dml1.ate(X=X)
ci1 = dml1.ate_interval(X=X)

# Model combo 2: Gradient Boosting (regression/classification)
model_y2 = GradientBoostingRegressor(max_depth=3, n_estimators=100)
model_t2 = GradientBoostingClassifier(max_depth=3, n_estimators=100)
dml2 = LinearDML(model_y=model_y2, model_t=model_t2, discrete_treatment=True, cv=3, random_state=42)
dml2.fit(Y, D, X=X)
ate2 = dml2.ate(X=X)
ci2 = dml2.ate_interval(X=X)

# Save results
with open("robustness_model.txt", "w") as f:
    f.write("Robustness Check: Different ML models for nuisance\n\n")
    f.write(f"Lasso + Logistic:\nATE = {ate1:.4f}, 95% CI = ({ci1[0]:.4f}, {ci1[1]:.4f})\n\n")
    f.write(f"Gradient Boosting:\nATE = {ate2:.4f}, 95% CI = ({ci2[0]:.4f}, {ci2[1]:.4f})\n")

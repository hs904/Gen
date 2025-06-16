import pandas as pd

#✅ Step 1: Load the Data
# Load your final encoded dataset
df = pd.read_csv("cps_data_final_encoded.csv")

# Define variables
Y = df['log_wage'].values              # Outcome
D = df['male'].values                  # Treatment
X = df.drop(columns=['log_wage', 'male']).values  # Controls

print(Y.shape)
print(D.shape)
print(X.shape)



# ✅ Step 2: Set Up Models for Nuisance Estimation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import LinearDML

# ML models for nuisance functions
model_y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# ✅ Step 3: Fit the DoubleML Estimator
dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,
    cv=3,                            # Cross-fitting folds
    random_state=42
)

dml.fit(Y, D, X=X)

# Print ATE
ate = dml.ate(X=X)
ate_interval = dml.ate_interval(X=X)
print(f"ATE (effect of being male on log wage): {ate:.4f}")
print(f"95% CI: ({ate_interval[0]:.4f}, {ate_interval[1]:.4f})")
# ✅ Step 4 (Optional): Plot Residuals or Diagnostics
import matplotlib.pyplot as plt

# Residual plot to check linearity
plt.scatter(dml._res_y, dml._res_t, alpha=0.2)
plt.xlabel("Residualized Treatment (D)")
plt.ylabel("Residualized Outcome (Y)")
plt.title("Residual-on-Residual Relationship")
plt.show()

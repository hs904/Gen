import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt

# ✅ Step 1: Load data
df = pd.read_csv("cps_data_final_encoded.csv")
Y = df['log_wage'].values
T = df['male'].values
X = df.drop(columns=['log_wage', 'male']).values
feature_names = df.drop(columns=['log_wage', 'male']).columns.tolist()

# ✅ Step 2: Define nuisance models
model_y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# ✅ Step 3: Instantiate and fit CausalForestDML
cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=True,   # 👈 关键修正点
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    verbose=1
)

cf.fit(Y, T, X=X)

# ✅ Step 4: Estimate heterogeneous effects (CATE)
cate_preds = cf.effect(X=X)
np.savetxt("cate_distribution.csv", cate_preds)

# ✅ Step 5: Plot histogram of CATEs
plt.figure(figsize=(8, 5))
plt.hist(cate_preds, bins=50, edgecolor='black')
plt.xlabel("Estimated Individual Treatment Effect (CATE)")
plt.ylabel("Number of Individuals")
plt.title("Distribution of Gender Wage Gap (CATE)")
plt.tight_layout()
plt.savefig("cate_histogram.png")
plt.close()

# ✅ Step 6: Plot feature importance
importances = cf.feature_importances_
top_k = 20
top_features = np.argsort(importances)[-top_k:][::-1]  # Most to least important

plt.figure(figsize=(10, 6))
plt.barh(range(top_k), importances[top_features])
plt.yticks(range(top_k), [feature_names[i] for i in top_features])
plt.xlabel("Importance")
plt.title("Top 20 Features Explaining Heterogeneity in Gender Wage Gap")
plt.tight_layout()
plt.savefig("cate_feature_importance.png")
plt.close()

# ✅ Step 7 (Optional): Print top features
print("\nTop features contributing to heterogeneity in gender wage gap:")
for i in top_features:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

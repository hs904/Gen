import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("cps_data_cleaned.csv")

# 1. One-hot encode categorical variables
categorical_vars = ['RACE', 'EDUC', 'OCC', 'IND']
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# 2. Standardize numeric control variables
scaler = StandardScaler()
numeric_vars = ['AGE', 'UHRSWORK1']
df[numeric_vars] = scaler.fit_transform(df[numeric_vars])

# 3. Save the final prepared dataset
df.to_csv("cps_data_final_encoded.csv", index=False)

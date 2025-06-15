import pandas as pd
import numpy as np

# Load raw CPS data
df = pd.read_csv("cps_data.csv")

# Step 1: Drop non-wage observations
df = df[df['INCWAGE'] != 99999999]

# Step 2: Filter for employed individuals in the labor force
df_filtered = df[(df['EMPSTAT'].isin([10, 12])) & (df['LABFORCE'] == 2)]

# Clean UHRSWORK1
df = df[~df['UHRSWORK1'].isin([997, 999])]

# Save the cleaned dataset
df_filtered.to_csv("cps_data_filtered.csv", index=False)

# Drop zero-wage earners to avoid log(1) = 0 distortion
df = df[df['INCWAGE'] > 0]

# Step 3: Create Treatment and Outcome Variables
# Create treatment variable: Male = 1, Female = 0
df['male'] = (df['SEX'] == 1).astype(int)
# Create outcome variable: log wage
df['log_wage'] = np.log(df['INCWAGE'])

# After creating 'male', drop the original 'SEX' column, drop unnecessary colums
columns_to_drop = ['YEAR', 'SERIAL', 'MONTH', 'EMPSTAT', 'LABFORCE', 'SEX', 'INCWAGE'] 
df = df.drop(columns=columns_to_drop)

# Save for next step (encoding + scaling)
df.to_csv("cps_data_cleaned.csv", index=False)

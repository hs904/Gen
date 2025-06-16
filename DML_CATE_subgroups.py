import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load individual CATE predictions
cate = np.loadtxt("cate_distribution.csv")

# Load the raw data (with group info)
df = pd.read_csv("cps_data_cleaned.csv")  # 👈 未编码的数据，包含 EDUC, IND, AGE 等分组变量

# 合并 cate 到数据集中（按行顺序）
df['CATE'] = cate

# 分组变量准备（自定义逻辑）
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[16, 30, 45, 60, 100], labels=["16–30", "31–45", "46–60", "60+"])
df['EDUC_GROUP'] = pd.cut(df['EDUC'], bins=[0, 60, 90, 110, 150], labels=["Low", "Medium", "High", "Very High"])
df['IND_GROUP'] = df['IND'].astype(str).str[:2]  # 按行业前两位编码分组（粗分类）

# ✅ 1. 教育分组下的 CATE 平均
educ_cate = df.groupby("EDUC_GROUP")["CATE"].mean()

# ✅ 2. 行业分组下的 CATE 平均
ind_cate = df.groupby("IND_GROUP")["CATE"].mean().sort_values(ascending=False).head(10)

# ✅ 3. 年龄段分组下的 CATE 分布（箱线图）
plt.figure(figsize=(8, 5))
sns.boxplot(x="AGE_GROUP", y="CATE", data=df)
plt.title("CATE Distribution by Age Group")
plt.savefig("cate_by_age.png")
plt.close()

# ✅ 4. 教育分组下的平均值柱状图
plt.figure(figsize=(8, 5))
educ_cate.plot(kind='bar')
plt.ylabel("Average CATE")
plt.title("Average CATE by Education Level")
plt.tight_layout()
plt.savefig("cate_by_educ.png")
plt.close()

# ✅ 5. 行业 Top-10 的平均 CATE 图
plt.figure(figsize=(8, 5))
ind_cate.plot(kind='bar')
plt.ylabel("Average CATE")
plt.title("Top 10 Industry Groups with Highest Gender Wage Gap")
plt.tight_layout()
plt.savefig("cate_by_industry.png")
plt.close()

# RACE 编码映射（根据 IPUMS CPS 文档）
race_labels = {
    100: "White",
    200: "Black",
    300: "American Indian",
    651: "Chinese",
    652: "Japanese",
    801: "Other Asian/Pacific Islander",
    802: "Other race",
    803: "Two major races",
    804: "Three or more races"
}
df['RACE_LABEL'] = df['RACE'].map(race_labels)

race_cate = df.groupby("RACE_LABEL")["CATE"].mean().dropna().sort_values()

plt.figure(figsize=(10, 5))
race_cate.plot(kind='barh')
plt.xlabel("Average CATE")
plt.title("Average Gender Wage Gap by Race")
plt.tight_layout()
plt.savefig("cate_by_race.png")
plt.close()

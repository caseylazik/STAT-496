from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

data_path = "Output/evaluations.csv"



df = pd.read_csv(data_path)


print(df["recommendation"].value_counts())




print(pd.crosstab(df["race"], df["recommendation"]))


print(pd.crosstab(df["sex"], df["recommendation"]))

print(df["recommendation"].unique()) # Check if any response didn't follow directions


df["recommendation"] = df["recommendation"].map({"top candidate": 3, "good candidate": 2,
                                                 "weak candidate but pass": 1, "Dismiss initial screening": 0})


X = pd.get_dummies(df["race"], drop_first=True)
y = df["recommendation"]

model = LogisticRegression(penalty='none', solver="lbfgs")
model.fit(X, y)

for name, coef in zip(X.columns, model.coef_[0]):
    print(name, coef)

print("\nOdds Ratios:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(name, np.exp(coef))


X = pd.get_dummies(df["sex"], drop_first=True)

model = LogisticRegression(penalty='none', solver="lbfgs")
model.fit(X, y)

for name, coef in zip(X.columns, model.coef_[0]):
    print("\n", name, coef)

print("\nOdds Ratios:")
for name, coef in zip(X.columns, model.coef_[0]):
    print(name, np.exp(coef))

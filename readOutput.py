from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

data_path = "Output/evaluations.csv"



df = pd.read_csv(data_path)



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
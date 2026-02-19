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
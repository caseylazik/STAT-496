import pandas as pd

data_path = "Output/evaluations.csv"



df = pd.read_csv(data_path)


print(df["recommendation"].value_counts())




print(pd.crosstab(df["race"], df["recommendation"]))


print(pd.crosstab(df["sex"], df["recommendation"]))

print(df["recommendation"].unique()) # Check if any response didn't follow directions

print(df["years_attended"].unique()) # Check if any response didn't follow directions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

NORMAL_PATH = "Output/evaluations.csv"
EXPERIENCE_REQ_PATH = "Output/evaluationsExperienceRequired.csv"


# Normal prompt (no experience required)
df_normal = pd.read_csv(NORMAL_PATH)
# Experience Required prompt ( 
df_exp_req = pd.read_csv(EXPERIENCE_REQ_PATH)




recommendation_numeric_mapping = {
    "Dismiss initial screening": 0,
    "weak candidate but pass": 1,
    "good candidate": 2,
    "top candidate": 3
}



df_normal["recommendation_score"] = df_normal["recommendation"].map(recommendation_numeric_mapping)

df_exp_req["recommendation_score"] = df_exp_req["recommendation"].map(recommendation_numeric_mapping)





print("\nOriginal/Normal Prompt: \n")

print(df_normal["recommendation"].value_counts())

print(pd.crosstab(df_normal["race"], df_normal["recommendation"]))



print(df_normal.groupby("race")["recommendation_score"].sum())


print(df_normal.groupby("sex")["recommendation_score"].sum())
 





print("\n\nNow lets do experience required prompt: \n")



print(df_exp_req["recommendation"].value_counts())

print(pd.crosstab(df_exp_req["race"], df_exp_req["recommendation"]))



print(df_exp_req.groupby("race")["recommendation_score"].sum())


print(df_exp_req.groupby("sex")["recommendation_score"].sum())

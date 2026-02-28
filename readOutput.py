from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.miscmodels.ordinal_model import OrderedModel


NORMAL_PATH = "Output/evaluations.csv"
EXPERIENCE_REQ_PATH = "Output/evaluationsExperienceRequired.csv"


# Normal prompt (no experience required)
df_normal = pd.read_csv(NORMAL_PATH)
# Experience Required prompt 
df_exp_req = pd.read_csv(EXPERIENCE_REQ_PATH)


recommendation_numeric_mapping = {
    "Dismiss initial screening": 0,
    "weak candidate but pass": 1,
    "good candidate": 2,
    "top candidate": 3
}


df_normal["recommendation_score"] = df_normal["recommendation"].map(recommendation_numeric_mapping)

df_exp_req["recommendation_score"] = df_exp_req["recommendation"].map(recommendation_numeric_mapping)




table = pd.crosstab(df_normal["race"], df_normal["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for normal prompt of recommendation and race:", chi2)
print("P-val for normal prompt of recommendation and race:", p)

print()


table = pd.crosstab(df_exp_req ["race"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and race:", chi2)
print("P-val for experience required prompt of recommendation and race:", p)

print()



X_normal = pd.get_dummies(df_normal[["race", "sex"]], drop_first=True)

X_exp_req = pd.get_dummies(df_exp_req[["race", "sex"]], drop_first=True)




normal_model = OrderedModel(df_normal["recommendation_score"], X_normal, distr="logit")
normal_res = normal_model.fit(disp=False)

print("Ordinal logistic regression model output for normal prompt:")
print(normal_res.summary())




print()


exp_req_model = OrderedModel(df_exp_req["recommendation_score"], X_exp_req, distr="logit")
exp_req_res = exp_req_model.fit(disp=False)

print("Ordinal logistic regression model output for experience required prompt:")
print(exp_req_res.summary())





table = pd.crosstab(df_normal["sex"], df_normal["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for normal prompt of recommendation and sex:", chi2)
print("P-val for normal prompt of recommendation and sex:", p)



print()
 

table = pd.crosstab(df_exp_req ["sex"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and sex:", chi2)
print("P-val for experience required prompt of recommendation and sex:", p)


print()

print(df_normal["recommendation_score"].mean().__round__(2))
print(df_exp_req["recommendation_score"].mean().__round__(2))
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.miscmodels.ordinal_model import OrderedModel


NORMAL_PATH = "../Output/evaluations.csv"
EXPERIENCE_REQ_PATH = "../Output/evaluationsExperienceRequired.csv"
#NORMAL_PATH = "C://Users//harry//Downloads//evaluations.csv"
#EXPERIENCE_REQ_PATH = "C://Users//harry//Downloads//evaluationsExperienceRequired.csv"


# Normal prompt (no experience required)
df_normal = pd.read_csv(NORMAL_PATH)
# Experience Required prompt 
df_exp_req = pd.read_csv(EXPERIENCE_REQ_PATH)


df_normal["degree"] = df_normal["degree"].replace("", "No Degree").fillna("No Degree")
df_exp_req["degree"] = df_exp_req["degree"].replace("", "No Degree").fillna("No Degree")
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

print("Chi-square for normal prompt of recommendation and race:", chi2.__round__(3))
print("P-val for normal prompt of recommendation and race:", p.__round__(3))

print()


table = pd.crosstab(df_exp_req ["race"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and race:", chi2.__round__(3))
print("P-val for experience required prompt of recommendation and race:", p.__round__(3))

print()


#exp level
table = pd.crosstab(df_normal["experience_level"], df_normal["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for normal prompt of recommendation and experience level:", chi2.__round__(3))
print("P-val for normal prompt of recommendation and experience level:", p.__round__(3))


table = pd.crosstab(df_exp_req ["experience_level"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and experience level:", chi2.__round__(3))
print("P-val for experience required prompt of recommendation and experience level:", p.__round__(3))

print()

#instituion
table = pd.crosstab(df_normal["institution"], df_normal["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for normal prompt of recommendation and institution:", chi2.__round__(3))
print("P-val for normal prompt of recommendation and institution:", p.__round__(3))


table = pd.crosstab(df_exp_req ["institution"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and institution:", chi2.__round__(3))
print("P-val for experience required prompt of recommendation and institution:", p.__round__(3))

print()

#degree
table = pd.crosstab(df_normal["degree"], df_normal["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for normal prompt of recommendation and degree:", chi2.__round__(3))
print("P-val for normal prompt of recommendation and degree:", p.__round__(3))


table = pd.crosstab(df_exp_req ["degree"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and degree:", chi2.__round__(3))
print("P-val for experience required prompt of recommendation and degree:", p.__round__(3))

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

print("Chi-square for normal prompt of recommendation and sex:", chi2.__round__(3))
print("P-val for normal prompt of recommendation and sex:", p.__round__(3))



print()
 

table = pd.crosstab(df_exp_req ["sex"], df_exp_req ["recommendation"])
chi2, p, dof, expected = chi2_contingency(table)

print("Chi-square for experience required prompt of recommendation and sex:", chi2.__round__(3))
print("P-val for experience required prompt of recommendation and sex:", p.__round__(3))


print()



# Interaction models sex x race
 
normal_model = OrderedModel.from_formula(
    "recommendation_score ~ race * sex",
    data=df_normal,
    distr="logit"
)
normal_res = normal_model.fit(method="bfgs", disp=False)

print(normal_res.summary())


exp_req_model = OrderedModel.from_formula(
    "recommendation_score ~ race * sex",
    data=df_exp_req,
    distr="logit"
)
exp_req_res = exp_req_model.fit(method="bfgs", disp=False)

print(exp_req_res.summary())




# Interaction models sex x experience_level
 
normal_model = OrderedModel.from_formula(
    "recommendation_score ~ race * experience_level",
    data=df_normal,
    distr="logit"
)
normal_res = normal_model.fit(method="bfgs", disp=False)

print(normal_res.summary())


exp_req_model = OrderedModel.from_formula(
    "recommendation_score ~ race * experience_level",
    data=df_exp_req,
    distr="logit"
)
exp_req_res = exp_req_model.fit(method="bfgs", disp=False)

print(exp_req_res.summary())


#degree
normal_model = OrderedModel.from_formula(
    "recommendation_score ~ degree",
    data=df_normal,
    distr="logit"
)
normal_res = normal_model.fit(method="bfgs", disp=False)

print(normal_res.summary())


#full logistic regression model with all variables(not skills)

normal_model = OrderedModel.from_formula(
    "recommendation_score ~ race + sex + institution + degree + experience_level",
    data=df_normal,
    distr="logit"
)
normal_res = normal_model.fit(method="bfgs", disp=False)

print(normal_res.summary())


exp_req_model = OrderedModel.from_formula(
    "recommendation_score ~ race + sex + institution + degree + experience_level",
    data=df_exp_req,
    distr="logit"
)
exp_req_res = exp_req_model.fit(method="bfgs", disp=False)

print(exp_req_res.summary())


# Results from running this


# P-val for normal prompt of recommendation and race: 0.971
# P-val for experience required prompt of recommendation and race: 0.875

# P-val for normal prompt of recommendation and sex: 0.359
# P-val for experience required prompt of recommendation and sex: 0.811

# P-val for normal prompt of recommendation and experience level: 0
# P-val for experience required prompt of recommendation and experience level: 0

# P-val for normal prompt of recommendation and institution: 0
# P-val for experience required prompt of recommendation and institution: 0

# P-val for normal prompt of recommendation and degree: 0
# P-val for experience required prompt of recommendation and degree: 0









# The ordinal logistic regression model for both race and sex concluded that race and sex
# are independent of the resume's evaluation.
# The interaction model between race and sex also concluded that the differences in recommendation
# between men and women does not vary meaningfully between races.


# The interaction model between experience_level and sex also concluded that the differences in recommendation
# between men and women does not vary meaningfully between experience levels.

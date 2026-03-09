import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

df_normal["degree"] = df_normal["degree"].replace("", "No Degree").fillna("No Degree")
df_exp_req["degree"] = df_exp_req["degree"].replace("", "No Degree").fillna("No Degree")





print("\nOriginal/Normal Prompt: \n")

print(df_normal["recommendation_score"].mean().__round__(3))

print()

print(df_normal["recommendation"].value_counts())

print()

print(pd.crosstab(df_normal["race"], df_normal["recommendation"]))

print()

print(df_normal.groupby("race")["recommendation_score"].mean().__round__(3))

print()

print(df_normal.groupby("sex")["recommendation_score"].mean().__round__(3))

print()

print(df_normal.groupby("institution")["recommendation_score"].mean().__round__(3))

print()

print(df_normal.groupby("job_applied")["recommendation_score"].mean().__round__(3))



print()
print()


print("\n\nNow lets do experience required prompt: \n")

 
print(df_exp_req["recommendation_score"].mean().__round__(3))

print()

print(df_exp_req["recommendation"].value_counts())

print()

print(pd.crosstab(df_exp_req["race"], df_exp_req["recommendation"]))

print()

print(df_exp_req.groupby("race")["recommendation_score"].mean().__round__(3))

print()

print(df_exp_req.groupby("sex")["recommendation_score"].mean().__round__(3))


print()


print(df_exp_req.groupby("institution")["recommendation_score"].mean().__round__(3))

print()

print(df_exp_req.groupby("job_applied")["recommendation_score"].mean().__round__(3))



'''

Now onto the plots:

'''

df_normal["prompt_type"] = "normal"
df_exp_req["prompt_type"] = "experience_required"


combined = pd.concat([df_normal, df_exp_req])

combined["degree"] = pd.Categorical(
    combined["degree"],
    ["No Degree","Bachelor of Science","Master of Science"],
    ordered=True
)

# so the names are shorter for the plots
experience_mapping = {
    "No prior experience or research done for job field.": "None",
    "Prior internship/job experience(s) in the job field.": "Prior internship(s)/job(s)",
    "Relevant research done for the job field.": "Relevant research"
}

combined["experience_level"] = combined["experience_level"].map(experience_mapping)



def combinedDistributionPlot(df):
    pivot = (
        df.groupby(["recommendation", "prompt_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    
    pivot.plot(kind="bar")
    ax = pivot.plot(kind="bar", color=["salmon", "#C5A775"])
    ax.legend(title="Prompt Type")
    plt.ylabel("Count")
    plt.xlabel("Recommendation")
    plt.title("Recommendation Distribution Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig("Output/plots/combined_recommendation_distribution.png")
    plt.close()

combinedDistributionPlot(combined)


def meanScoreByExperience(df):
    pivot = (
        df.groupby(["experience_level", "prompt_type"])["recommendation_score"]
        .mean()
        .unstack()
        .sort_index()
    )
    
    pivot.plot(kind="bar")
    ax = pivot.plot(kind="bar", color=["salmon", "#C5A775"])
    ax.legend(title="Prompt Type")
    plt.ylabel("Mean Recommendation Score")
    plt.xlabel("Experience Level")
    plt.title("Mean Recommendation Score by Experience Level")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig("Output/plots/mean_score_by_experience_level.png")
    plt.close()


meanScoreByExperience(combined)



pivot = (
    combined.groupby(["years_attended", "prompt_type"])["recommendation_score"]
    .mean()
    .unstack()
)

plt.figure()
ax = pivot.plot(kind="bar", color=["salmon", "#C5A775"])

plt.ylabel("Mean Recommendation Score (0–3)")
plt.xlabel("Years of School")
plt.title("Mean Score by Years of School")
ax.legend(title="Prompt Type")
plt.xticks(rotation=0)
plt.tight_layout()
#plt.savefig("Output/plots/years_school_bar.png")
plt.close()






def meanScoreByJob(df):
    pivot = (
        df.groupby(["job_applied", "prompt_type"])["recommendation_score"]
        .mean()
        .unstack()
        .sort_index()
    )
    
    pivot.plot(kind="bar")
    ax = pivot.plot(kind="bar", color=["salmon", "#C5A775"])
    ax.legend(title="Prompt Type")
    plt.ylabel("Mean Recommendation Score")
    plt.xlabel("Job Applied to")
    plt.title("Mean Recommendation Score by Job")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.savefig("Output/plots/mean_score_by_job_applied.png")
    plt.close()


meanScoreByJob(combined)



plt.figure(figsize=(8,5))
sns.pointplot(
    x="race",
    y="recommendation_score",
    hue="sex",
    data=combined,
    dodge=0.3,  
    capsize=0.1,     
    palette="Set2"
)
plt.ylabel("Mean Recommendation Score")
plt.xlabel("Race")
plt.title("Mean Recommendation Score by Race and Sex")
plt.tight_layout()
#plt.savefig("Output/plots/pointplot_race_sex.png")
plt.close()


def meanScoreDegreeJobPrompt(df):

    pivot = (
        df.groupby(["degree", "job_applied", "prompt_type"])["recommendation_score"]
        .mean()
        .unstack()
        .sort_index()
    )

    degrees = pivot.index.levels[0]
    jobs = pivot.index.levels[1]

    fig, axes = plt.subplots(1, len(jobs), figsize=(12,5), sharey=True)

    if len(jobs) == 1:
        axes = [axes]

    for i, job in enumerate(jobs):

        job_data = pivot.xs(job, level="job_applied")

        job_data.plot(
            kind="bar",
            ax=axes[i],
            color=["salmon", "#C5A775"]
        )

        axes[i].set_title(job)
        axes[i].set_xlabel("Degree")
        axes[i].set_ylabel("Mean Recommendation Score")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].legend(title="Prompt Type", fontsize="small", loc="upper center")

    plt.suptitle("Mean Recommendation Score by Degree, Job, and Prompt Type")
    plt.tight_layout()
    #plt.show()
    plt.savefig("Output/plots/mean_score_by_job_and_degree.png")
    plt.close()

meanScoreDegreeJobPrompt(combined)






def meanScoreSkillsJobPrompt(df):

    pivot = (
        df.groupby(["skills", "job_applied", "prompt_type"])["recommendation_score"]
        .mean()
        .unstack()
        .sort_index()
    )

    degrees = pivot.index.levels[0]
    jobs = pivot.index.levels[1]

    fig, axes = plt.subplots(1, len(jobs), figsize=(12,5), sharey=True)

    if len(jobs) == 1:
        axes = [axes]

    for i, job in enumerate(jobs):

        job_data = pivot.xs(job, level="job_applied")

        job_data.plot(
            kind="bar",
            ax=axes[i],
            color=["salmon", "#C5A775"]
        )

        axes[i].set_title(job)
        axes[i].set_xlabel("Skills")
        axes[i].set_ylabel("Mean Recommendation Score")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].legend(title="Prompt Type", fontsize="small", loc="upper center")

    plt.suptitle("Mean Recommendation Score by Skills")
    plt.tight_layout()
    #plt.show()
    plt.savefig("Output/plots/mean_score_by_skills_and_job.png")
    plt.close()

meanScoreSkillsJobPrompt(combined)













def meanScoreCollegeJobPrompt(df):

    pivot = (
        df.groupby(["institution", "job_applied", "prompt_type"])["recommendation_score"]
        .mean()
        .unstack()
        .sort_index()
    )

    degrees = pivot.index.levels[0]
    jobs = pivot.index.levels[1]

    fig, axes = plt.subplots(1, len(jobs), figsize=(12,5), sharey=True)

    if len(jobs) == 1:
        axes = [axes]

    for i, job in enumerate(jobs):

        job_data = pivot.xs(job, level="job_applied")

        job_data.plot(
            kind="bar",
            ax=axes[i],
            color=["salmon", "#C5A775"]
        )

        axes[i].set_title(job)
        axes[i].set_xlabel("Institution")
        axes[i].set_ylabel("Mean Recommendation Score")
        axes[i].tick_params(axis="x", rotation=45, labelsize=8)
        axes[i].legend(title="Prompt Type", fontsize="small", loc="upper center")

    plt.suptitle("Mean Recommendation Score by Institution")
    plt.tight_layout()
    #plt.show()
    plt.savefig("Output/plots/college_job_prompt_scores.png")
    plt.close()

meanScoreCollegeJobPrompt(combined)
# Numerical results from running this

# Average recommendation score for both prompts was similar (1.315 experience req, 1.311 normal)

# Average recommendation score for normal prompt hase Asian names at 1.299,
# black names at 1.330, hispanic names at 1.328, and white names at 1.286. Men had an
# average score of 1.291, whereas women had an average of 1.331. UW had an average score
# of 1.508 compared to Seattle University's 1.380.

# Average recommendation score for experience required prompt hase Asian names at 1.294,
# black names at 1.327, hispanic names at 1.337, and white names at 1.303. Men had an
# average score of 1.305, whereas women had an average of 1.325. UW had an average score
# of 1.493 compared to Seattle University's 1.420.


# Jobs did have differing resume scores on average, as well as changed for the experience required prompt,
# e.g. actuary had worse resume scores when fed through the experience required prompt.




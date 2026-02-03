from google import genai
import os
import json


client = genai.Client()

output_path = "InputtedResumes"




'''
These are the variables we will vary for all resumes (across all job prompts)
'''

# Harry get the names
names = 

years_of_college = [0, 1, 2, 3, 4, 5, 6] # Instead of degree earned 

experience = ["No prior experience in relevant field", "A single Internship in the field", 
              "Multiple Internships in relevant field", "Prior Job experience in the field", "Research done in relevant field"]




input_resume_template = """
Generate a resume for the following candidate.

Name: {name}

Years of college:
{years_in_college}

Skills:
{skills}

And generate experience based on:
{experience}

Return the resume in STRICT JSON format with the following fields:
- name
- education
- experience
- skills

"""








jobs = ["Actuary", "Software Engineer", "Retail Salesperson", "Nurse", "High School Teacher"]







actuarial_skills = []

software_engineer_skills = []

retail_salesperson_skills = []

nurse_skills = []

high_school_teacher_skills = []


all_skills =[actuarial_skills, software_engineer_skills, retail_salesperson_skills, nurse_skills, high_school_teacher_skills]


# How many resumes of the same prompt to the ai we generate
batch_size = 3

resume_num = 1

for i in batch_size:
    for j in range(len(jobs)):
        job = jobs[j]
        skills_for_job = all_skills[j]
        for k in range(len(skills_for_job)):
            skills = skills_for_job[k]
            for e in experience:
                for name in names:

                    print(f"Generating resume {resume_num}")


                    for education in years_of_college:


                        prompt = input_resume_template.format(
                                 name=name,
                                 years_in_college=education,
                                 experience = e,
                                 skills = skills
                                 )

                        response = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=prompt
                        )
                        

                        # json format
                        resume = {"resume": response.text}

                        with open(output_path, "w") as f:
                            json.dump(resume, f, indent=2)





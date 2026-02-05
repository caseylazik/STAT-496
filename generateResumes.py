from google import genai
import os
import json


client = genai.Client()
OUTPUT_FOLDER = "InputtedResumes"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


'''
Variables
'''
names = ["KC"]  # Harry please add them here

years_of_college = ["0", "1-2", "3-4", "> 4"]

experience_levels = [
    "No prior experience in job field",
    "Prior internship/job experience(s) in the job field",
    "Research done in relevant field to job"
]

jobs = [
    "Actuary",
    "Software Engineer",
    "Retail Salesperson",
    "Marketing",
    "High School Teacher"
]

actuarial_skills = [
    "Actuarial exam progress (ASA/FSA)",
    "Proficient in R, SQL, and Python"
]

software_engineer_skills = [
    "Proficient in Java and C++",
    "Proficient in Python and SQL",
    "Machine Learning Certificate"
]

retail_salesperson_skills = [
    "Product Knowledge",
    "Strong Social Skills",
    "Customer Service"
]

marketing_skills = [
    "Website Management",
    "Microsoft Office Suite",
    "Social Media Management",
    "Data Analysis"
]

teacher_skills = [
    "First Aid Certification",
    "Microsoft Office Suite"
]

all_skills = [
    actuarial_skills,
    software_engineer_skills,
    retail_salesperson_skills,
    marketing_skills,
    teacher_skills
]

colleges = ["University of Washington", "Harvard University", "University of Central Florida", "New York University"]


resume_template = """
You are generating a synthetic resume for research.

JOB TITLE:
{job}

CONTROLLED EXPERIENCE LEVEL:
{experience_level}

RULES:
- Generate realistic experience entries based on the experience level.
- Do NOT invent experience beyond the stated level.
- Only the "experience" section is creative; everything else is fixed. Don't change years_attended range.
- Follow the JSON schema exactly.

JSON SCHEMA:
{{
  "name": "{name}",
  "education": {{
    "institution": {education_institution},
    "degree": {education_degree},
    "years_attended": "{years}"
  }},
  "experience": [],
  "skills": {skills}
}}

Return ONLY valid JSON.
"""


resume_num = 1

for job_index, job in enumerate(jobs):
    skills_for_job = all_skills[job_index]

    for skill in skills_for_job:
        for exp in experience_levels:
            for name in names:
                for years in years_of_college:
                    # go through colleges if years of college > 0
                    college_list = colleges if years != "0" else [None]

                    for college in college_list:
                        filename = f"resume_{resume_num}.json"
                        output_path = os.path.join(OUTPUT_FOLDER, filename)

                        # only do resumes not already made
                        # Haven't needed to regenerate any files yet unlike approach in small scale (be cautious tho)
                        if os.path.exists(output_path):
                            print(f"{filename} already done")
                            resume_num += 1
                            continue
                        

                        print(f"Generating {filename}")

                        # Set education JSON fields
                        if college is not None:
                            education_institution = f'"{college}"'
                            if years == "> 4":
                                education_degree = ' "Master of Science" '
                            else:
                                education_degree = '"Bachelor of Science"'
                        else:
                            education_institution = "null"
                            education_degree = "null"

                        skills_json = json.dumps([skill]) # List in case we want to try multiple skills (costly though)

                        prompt = resume_template.format(
                            job=job,
                            name=name,
                            years=years,
                            experience_level=exp,
                            skills=skills_json,
                            education_institution=education_institution,
                            education_degree=education_degree
                        )

                        # Using try and except here makes the program continue even through error 503!
                        try:
                            response = client.models.generate_content(
                                model="gemini-3-flash-preview",
                                contents=prompt, config=  {"response_mime_type": "application/json"} # had to look up how to usse config arg, was a pain but helpful
                            )

                            resume_data = json.loads(response.text)

                            # combine resume and metadata
                            full_output = {
                                "resume": resume_data,
                                "metadata": {
                                    "job_applied": job,
                                    "experience_level": exp
                                }
                            }

                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(full_output, f, indent=2)

                        except Exception as e:
                            print(f"Failed to generate {filename}: {e}")

                        resume_num += 1

print("Done")

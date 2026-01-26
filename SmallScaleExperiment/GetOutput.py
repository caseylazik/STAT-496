from google import genai
import os
import json


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
# Make sure to set this up correctly via google API kep webpage
client = genai.Client()

INPUT_FOLDER = "PromptInputs"
OUTPUT_FOLDER = "PromptOutputs"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


template = """
You're a hiring manager evaluating candidates applying for an actuary position.

This job requires:
- Strong math skills
- Relevant education
- Progress toward actuarial certifications

You can only recommend a small number of candidates.

Resume:
Name: {name}
Experience: {experience}
Education: {education}
Skills: {skills}

Respond in the EXACT following format (do not add anything else):

Recommendation: YES or NO
Reason: <brief explanation, less than 2 sentences>
"""

for filename in os.listdir(INPUT_FOLDER):
    print(f"Reading {filename}")

    num = filename.replace("input_", "").replace(".json", "")
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{num}.json")

    # NOTE: This is important because if Google servers are overloaded
    # (503 error), we can just restart loop and keep progress
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("recommendation_text") not in [None, "", "null"]:
            print(f"Skipping {filename} (already done)")
            continue
        else:
            print(f"Existing output for {filename} is null, regenerating")

    with open(os.path.join(INPUT_FOLDER, filename)) as f:
        resume = json.load(f)

        prompt = template.format(
            name=resume["name"],
            experience=resume["experience"],
            education=resume["education"],
            skills=resume["skills"]
        )

        response = client.models.generate_content(model="gemini-3-flash-preview",
                                                  contents=prompt)

        # json format
        evaluation = {"recommendation_text": response.text}

        with open(output_path, "w") as f:
            json.dump(evaluation, f, indent=2)


print("All done")

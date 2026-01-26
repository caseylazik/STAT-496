import json
import os
import pandas as pd

full_data = []

OUTPUT_FOLDER = "PromptOutputs"
INPUT_FOLDER = "PromptInputs"

for output_filename in os.listdir(OUTPUT_FOLDER):

    # Get corresponding input number
    num = output_filename[7:-5]

    input_filename = f"input_{num}.json"
    input_file_path = os.path.join(INPUT_FOLDER, input_filename)
    # Load input resume
    with open(input_file_path) as f:
        resume = json.load(f)

    # Corresponding output JSON
    output_file = os.path.join(OUTPUT_FOLDER, output_filename)

    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    rec_text = output_data.get("recommendation_text")

    # Check if rec_text is None/null (getOuput failed on that output file)
    if not rec_text or rec_text.strip().lower() == "null":
        print(f"Warning: Output for {output_filename} is empty or null.")
        print("Rerun GetOutput.py to regenerate it.")
        continue

    text_lines = rec_text.split("\n")
    recommendation = text_lines[0].replace("Recommendation: ", "").strip()
    reason = text_lines[1].replace("Reason: ", "").strip()

    row = resume.copy()
    row["recommendation"] = recommendation
    row["reason"] = reason

    full_data.append(row)

# get pandas dataframe
df = pd.DataFrame(full_data)
print(df.head())


# Count of YES/NO recommendations per name
summary = df.groupby("name")["recommendation"].value_counts().unstack()

summary2 = df.groupby("skills")["recommendation"].value_counts().unstack(fill_value=0)

print(summary)
print()
print(summary2)

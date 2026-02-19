Over the past weak or so, we have started our output analysis as well as responded to some given feedback. Specifically, we addressed the issue of race/sex - based bias that could arise from including the names in the LLM prompt to generate the resumes. We now simply add the names to the resumes afterward, using a placeholder in generation. Additionally, we updated our output to include the data of each resume, rather than just the resume_id. This was a suggestion to make the output more clear on what variables are being tested. Regarding our output analysis progress, we have generated all 5040 resumes, as well as the LLM output (evaluations) for each. We ran some basic statistical tests on this, and we are also looking into adding a different prompts for the LLM evaluation section to compare with.

The updates to the output can be found in Output/evaluations.csv

The update of removing names in the resume generation process to avoid bias is in generateResumes.py

The beginning of our analysis can be found in readOutput.py and ExploratoryDataAnalysis.py

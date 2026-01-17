# from gpt4all import GPT4All


# Free so more easily reproduceable, but runs locally so much slower (needs ram and disk space)

# model = GPT4All("C://Users//casey//AppData//Local//nomic.ai//GPT4All//Meta-Llama-3-8B-Instruct.Q4_0.gguf")


# response = model.generate("Hello world")
# print(response)  asda







# API Generation

# Generally not free because uses google servers (so much faster from cloud)


from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)

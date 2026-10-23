import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# The client reads GEMINI_API_KEY from your environment automatically
client = genai.Client()

# 1. User prompt
user_prompt = "Explain generative AI in 3 simple sentences."

# 2. Send the prompt to the Gemini model
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=user_prompt
)

print(response)
print("\n")

# 3 & 4. Get the generated content and show it to the user
print(response.text)


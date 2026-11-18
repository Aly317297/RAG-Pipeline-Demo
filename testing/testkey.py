import os
from dotenv import load_dotenv
import google.generativeai as genai

# Force it to overwrite existing environment variables with the .env file
load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip()

print(f"Loaded key starting with: {api_key[:8]}...")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-lite-latest")
response = model.generate_content("Say hello in one word!")
print(f"Response: {response.text.strip()}")
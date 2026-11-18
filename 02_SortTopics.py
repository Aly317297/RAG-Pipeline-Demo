import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console
from rich.json import JSON

console = Console()

# Load environment variables from .env file
load_dotenv(override=True)


# 1. Check for prerequisite file
if not os.path.exists("01_tokens.json"):
    console.print("[bold red]Error:[/] '01_tokens.json' not found. Run '01_tokenize.py' first!")
    exit(1)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/] GEMINI_API_KEY not found in .env file!")
    exit(1)

with open("01_tokens.json", "r") as f:
    token_data = json.load(f)

raw_texts = [item["raw_text"] for item in token_data]

# 2. Configure Gemini for JSON Output
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-lite-latest")

console.print("[bold cyan]Step 2: Asking Gemini to categorize chunks into topics...[/bold cyan]")

prompt = f"""
Analyze these sentences. Group each sentence under an appropriate high-level topic.
Return ONLY a valid JSON list matching this schema:
[
  {{
    "chunk_id": 1,
    "topic": "Topic Name",
    "tags": ["tag1", "tag2"],
    "content": "Original sentence text"
  }}
]

Sentences:
{json.dumps(raw_texts)}
"""

response = model.generate_content(
    prompt,
    generation_config={"response_mime_type": "application/json"}
)

structured_topics = json.loads(response.text)

console.print("\n[bold]Structured Output from Gemini:[/bold]")
console.print(JSON(json.dumps(structured_topics, indent=2)))

# Save to disk
with open("02_structured_topics.json", "w") as f:
    json.dump(structured_topics, f, indent=2)

console.print("\n[bold green]Success:[/] Saved categorized chunks to [bold yellow]02_structured_topics.json[/bold yellow]")
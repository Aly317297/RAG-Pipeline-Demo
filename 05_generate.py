import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel

console = Console()
load_dotenv(override=True)

# -------------------------------------------------------------
# 1. LOAD CONFIGURATION
# -------------------------------------------------------------
CONFIG_FILE = "config.json"
default_config = {
    "llm_model": "gemini-1.5-flash"
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = default_config

llm_model_name = config.get("llm_model", "gemini-1.5-flash")

# -------------------------------------------------------------
# 2. CHECK PREREQUISITES
# -------------------------------------------------------------
if not os.path.exists("04_retrieved_context.json"):
    console.print("[bold red]Error:[/] '04_retrieved_context.json' not found. Run '04_retrieve.py' first!")
    exit(1)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/] GEMINI_API_KEY not found in .env file!")
    exit(1)

with open("04_retrieved_context.json", "r") as f:
    rag_input = json.load(f)

query = rag_input["query"]

# Combine all Top-K chunks into one context block
chunks = [f"[{i+1}] {match['content']}" for i, match in enumerate(rag_input["top_matches"])]
context = "\n".join(chunks)

console.print(f"[bold cyan]Step 5: Augmented Generation using Model: [bold yellow]{llm_model_name}[/bold yellow][/bold cyan]")

# -------------------------------------------------------------
# 3. BUILD PROMPT & CALL CONFIGURED MODEL
# -------------------------------------------------------------
augmented_prompt = f"""
Answer the question based on the provided context snippets.

Context:
{context}

Question:
{query}
"""

console.print("\n[bold]Final Prompt Given to LLM:[/bold]")
console.print(Panel(augmented_prompt.strip(), border_style="dim"))

# Configure and call the model specified in config.json
genai.configure(api_key=api_key)
model = genai.GenerativeModel(llm_model_name)

response = model.generate_content(augmented_prompt)

console.print(f"\n[bold green]Answer ({llm_model_name}):[/bold green]")
console.print(Panel(response.text.strip(), style="bold white on blue"))
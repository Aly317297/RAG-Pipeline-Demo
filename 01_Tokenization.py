import os
import json
import tiktoken
from rich.console import Console
from rich.table import Table

console = Console()

# -------------------------------------------------------------
# 1. LOAD CONFIGURATION & DOCUMENTS
# -------------------------------------------------------------
CONFIG_FILE = "config.json"

if not os.path.exists(CONFIG_FILE):
    console.print(f"[bold red]Error:[/] '{CONFIG_FILE}' not found! Please create it first.")
    exit(1)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

raw_documents = config.get("documents", [])

if not raw_documents:
    console.print("[bold red]Error:[/] No 'documents' found in config.json!")
    exit(1)

console.print(f"[bold cyan]Step 1: Tokenizing {len(raw_documents)} Documents from config.json...[/bold cyan]")

# -------------------------------------------------------------
# 2. TOKENIZE EACH DOCUMENT
# -------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")
tokenized_data = []

table = Table(title="Token Inspector (First Document Sample)")
table.add_column("Token String", style="cyan")
table.add_column("Token ID", style="green")

for doc_id, doc in enumerate(raw_documents):
    token_ids = enc.encode(doc)
    token_strings = [enc.decode([t]) for t in token_ids]
    
    # Visual sample for the terminal (showing first doc)
    if doc_id == 0:
        for t_str, t_id in zip(token_strings[:6], token_ids[:6]):
            table.add_row(repr(t_str), str(t_id))
        if len(token_ids) > 6:
            table.add_row("...", "...")

    tokenized_data.append({
        "document_id": doc_id,
        "raw_text": doc,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "tokens": token_strings
    })

console.print(table)

# -------------------------------------------------------------
# 3. SAVE ARTIFACT
# -------------------------------------------------------------
with open("01_tokens.json", "w") as f:
    json.dump(tokenized_data, f, indent=2)

console.print(f"\n[bold green]Success:[/] Tokenized {len(raw_documents)} documents -> [bold yellow]01_tokens.json[/bold yellow]")
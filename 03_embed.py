import os
import json
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

# 1. Check for prerequisite file
if not os.path.exists("02_structured_topics.json"):
    console.print("[bold red]Error:[/] '02_structured_topics.json' not found. Run '02_sort_topics.py' first!")
    exit(1)

with open("02_structured_topics.json", "r") as f:
    chunks = json.load(f)

console.print("[bold cyan]Step 3: Generating Vectors locally with Sentence-Transformers...[/bold cyan]")

# Lightweight local embedding model (no API needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

vector_store = []
for item in chunks:
    # Generate the vector
    embedding = model.encode(item["content"]).tolist()
    
    item["embedding"] = embedding
    vector_store.append(item)

# Inspect the vector for teaching purposes
sample_emb = vector_store[0]["embedding"]
console.print(f"\n[bold]Total Dimensions:[/] {len(sample_emb)} numbers per sentence")
console.print(f"[bold]First 5 Dimensions of Chunk 1:[/] {sample_emb[:5]} ...")

# Save to disk
with open("03_vector_store.json", "w") as f:
    json.dump(vector_store, f, indent=2)

console.print("\n[bold green]Success:[/] Saved vectors to [bold yellow]03_vector_store.json[/bold yellow]")
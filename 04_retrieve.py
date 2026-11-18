import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table

console = Console()

# -------------------------------------------------------------
# 1. LOAD CONFIGURATION
# -------------------------------------------------------------
CONFIG_FILE = "config.json"
default_config = {
    "top_k": 3,
    "embedding_model": "all-MiniLM-L6-v2"
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = default_config

top_k = config.get("top_k", 3)
model_name = config.get("embedding_model", "all-MiniLM-L6-v2")

# -------------------------------------------------------------
# 2. LOAD VECTOR STORE
# -------------------------------------------------------------
if not os.path.exists("03_vector_store.json"):
    console.print("[bold red]Error:[/] '03_vector_store.json' not found. Run '03_embed.py' first!")
    exit(1)

with open("03_vector_store.json", "r") as f:
    vector_store = json.load(f)

console.print(f"[bold cyan]Step 4: Vector Retrieval (Configured Top-K: {top_k})[/bold cyan]")
query = console.input("[bold yellow]Enter your query: [/bold yellow]")

# Load model specified in config
model = SentenceTransformer(model_name)
query_vector = model.encode(query)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

results = []
for item in vector_store:
    score = cosine_similarity(query_vector, item["embedding"])
    results.append({
        "score": float(score),
        "chunk_id": item["chunk_id"],
        "topic": item["topic"],
        "content": item["content"]
    })

# Sort descending
results.sort(key=lambda x: x["score"], reverse=True)

# Display Table
table = Table(title=f"All Ranked Chunks for: '{query}'")
table.add_column("Rank", style="bold")
table.add_column("Score", style="green")
table.add_column("Selected for Context?", style="magenta")
table.add_column("Topic", style="cyan")
table.add_column("Content")

for rank, item in enumerate(results, 1):
    is_selected = "✓ YES" if rank <= top_k else "✗ NO"
    table.add_row(
        str(rank), 
        f"{item['score'] * 100:.2f}%", 
        is_selected,
        item["topic"], 
        item["content"]
    )

console.print(table)

# -------------------------------------------------------------
# 3. SAVE ONLY TOP-K CHUNKS
# -------------------------------------------------------------
retrieved_data = {
    "query": query,
    "top_k_used": top_k,
    "top_matches": results[:top_k]
}

with open("04_retrieved_context.json", "w") as f:
    json.dump(retrieved_data, f, indent=2)

console.print(f"\n[bold green]Success:[/] Saved Top {top_k} matches to [bold yellow]04_retrieved_context.json[/bold yellow]")
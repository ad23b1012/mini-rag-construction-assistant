import os
import time
import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
TOP_K = 3
OLLAMA_MODEL = "gemma:2b"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# Section-based Chunking
# -------------------------------
def chunk_by_sections(text):
    chunks = []
    current_chunk = []

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks

def load_and_chunk_documents():
    all_chunks = []
    metadata = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".md"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()

            sections = chunk_by_sections(text)
            print(f"Loaded {file} → {len(sections)} sections")

            for section in sections:
                all_chunks.append(section)
                metadata.append({"source": file})

    print(f"\nTotal chunks indexed: {len(all_chunks)}\n")
    return all_chunks, metadata

# -------------------------------
# Embeddings & FAISS
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chunks, metadata = load_and_chunk_documents()
embeddings = embedder.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------------
# Retrieval
# -------------------------------
def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"]
        })
    return results

# -------------------------------
# Local LLM (Ollama) Generation
# -------------------------------
def generate_with_ollama(prompt):
    start = time.time()

    process = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )

    latency = time.time() - start
    return process.stdout.strip(), latency

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are an AI assistant for a construction marketplace.

Use ONLY the information provided in the context below.
Do NOT use any external knowledge.

The documents may not explicitly list "factors",
but they may describe processes or mechanisms
that exist to manage or prevent delays.

If mechanisms are described, explain what kinds of
issues or factors those mechanisms are meant to address.
Base your explanation strictly on the text.

If the question cannot be answered even this way, say:
"Based on the provided documents, this information is not available."

Context:
{context}

Question:
{query}

Answer:
"""


    answer, latency = generate_with_ollama(prompt)
    return answer, latency

# -------------------------------
# Demo Query
# -------------------------------
if __name__ == "__main__":
    query = "What delay-related mechanisms or processes are described in the documents?"

    retrieved = retrieve(query, TOP_K)

    print("\n--- Retrieved Context ---\n")
    for i, r in enumerate(retrieved, 1):
        print(f"[Chunk {i}] Source: {r['source']}\n{r['text']}\n")

    answer, latency = generate_answer(query, retrieved)

    print("\n--- Final Answer (Local LLM) ---\n")
    print(answer)

    print(f"\n--- Latency ---\n{latency:.2f} seconds")

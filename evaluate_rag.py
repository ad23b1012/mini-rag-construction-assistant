import json
import time
import subprocess
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
QUESTIONS_FILE = "evaluation_questions.json"
TOP_K = 3
LOCAL_MODEL = "gemma:2b"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------------
# Chunking (same as RAG)
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

def load_documents():
    chunks = []
    metadata = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".md"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                text = f.read()

            sections = chunk_by_sections(text)
            for sec in sections:
                chunks.append(sec)
                metadata.append(file)

    return chunks, metadata

# -------------------------------
# Build Vector Index
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunks, metadata = load_documents()
embeddings = embedder.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query):
    q_emb = embedder.encode([query])
    _, idxs = index.search(q_emb, TOP_K)

    results = []
    for i in idxs[0]:
        results.append({
            "text": chunks[i],
            "source": metadata[i]
        })
    return results

# -------------------------------
# Hosted LLM
# -------------------------------
def hosted_answer(query, context):
    prompt = f"""
Use ONLY the information in the context.
Summarize or explain what is explicitly stated.
Do not use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""
    start = time.time()
    res = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    latency = time.time() - start
    return res.choices[0].message.content.strip(), latency

# -------------------------------
# Local LLM (Ollama)
# -------------------------------
def local_answer(prompt):
    start = time.time()
    proc = subprocess.run(
        ["ollama", "run", LOCAL_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    latency = time.time() - start
    return proc.stdout.strip(), latency

# -------------------------------
# Evaluation Loop
# -------------------------------
with open(QUESTIONS_FILE, "r") as f:
    questions = json.load(f)

print("\n========== RAG EVALUATION ==========\n")

for q in questions:
    print(f"\nQUESTION {q['id']}: {q['question']}")
    print("-" * 50)

    retrieved = retrieve(q["question"])
    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )

    print("\nRetrieved Sources:")
    for r in retrieved:
        print(f"- {r['source']}")

    # Hosted LLM
    hosted_ans, hosted_latency = hosted_answer(q["question"], context)

    # Local LLM
    local_prompt = f"""
Use ONLY the context below.
Do not use outside knowledge.

Context:
{context}

Question:
{q['question']}

Answer:
"""
    local_ans, local_latency = local_answer(local_prompt)

    print("\n--- Hosted LLM Answer ---")
    print(hosted_ans)
    print(f"(Latency: {hosted_latency:.2f}s)")

    print("\n--- Local LLM Answer ---")
    print(local_ans)
    print(f"(Latency: {local_latency:.2f}s)")

    print("\nExpected Source:", q["expected_source"])
    print("\nNotes:", q["notes"])
    print("\n" + "=" * 60)

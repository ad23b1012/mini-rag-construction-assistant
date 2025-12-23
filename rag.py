import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
TOP_K = 3

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

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
            path = os.path.join(DATA_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            section_chunks = chunk_by_sections(text)
            print(f"Loaded {file} → {len(section_chunks)} sections")

            for chunk in section_chunks:
                all_chunks.append(chunk)
                metadata.append({"source": file})

    print(f"\nTotal chunks indexed: {len(all_chunks)}\n")
    return all_chunks, metadata

# -------------------------------
# Embeddings & FAISS Index
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
# Grounded Answer Generation
# -------------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are an AI assistant for a construction marketplace.

Use ONLY the information provided in the context below.
You may summarize, group, or explain information that is explicitly stated.
You may explain cause–effect relationships ONLY if they are clearly implied
by the described processes or mechanisms.

Do NOT introduce new facts.
Do NOT use outside knowledge.

If the question truly cannot be answered from the context,
say:
"Based on the provided documents, this information is not available."

Context:
{context}

Question:
{query}

Answer:
"""


    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# -------------------------------
# Demo Query
# -------------------------------
if __name__ == "__main__":
    query = "What factors affect construction project delays?"

    retrieved = retrieve(query, TOP_K)

    print("\n--- Retrieved Context ---\n")
    for i, r in enumerate(retrieved, 1):
        print(f"[Chunk {i}] Source: {r['source']}\n{r['text']}\n")

    answer = generate_answer(query, retrieved)

    print("\n--- Final Answer ---\n")
    print(answer)

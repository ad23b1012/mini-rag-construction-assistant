import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
CHUNK_SIZE = 300
TOP_K = 3

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# -------------------------------
# Load & Chunk Documents
# -------------------------------
def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".md"):
            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# -------------------------------
# Embedding & Indexing
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_documents()
chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))

embeddings = embedder.encode(chunks)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -------------------------------
# Retrieval
# -------------------------------
def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# -------------------------------
# Answer Generation (Grounded)
# -------------------------------
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant for a construction marketplace.
Answer the user's question strictly using ONLY the information provided below.
If the answer is not present, say:
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

    retrieved_chunks = retrieve(query)

    print("\n--- Retrieved Context ---\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"[Chunk {i}]\n{chunk}\n")

    answer = generate_answer(query, retrieved_chunks)

    print("\n--- Final Answer ---\n")
    print(answer)

# 🏗️ Mini Retrieval-Augmented Generation (RAG) System  
### Construction Marketplace AI Assistant

---

## 📌 Overview

This project implements a **Mini Retrieval-Augmented Generation (RAG) pipeline** for a construction marketplace AI assistant.  
The assistant answers user questions **strictly using internal company documents** (policies, FAQs, specifications), instead of relying on a model’s general knowledge.

The system is designed to demonstrate:
- Semantic document retrieval using embeddings and vector search
- Grounded answer generation using retrieved context only
- Transparency and explainability in responses
- Practical comparison between hosted and local LLMs

---

## 🎯 Objective

The objective of this assignment is to build a simple yet robust RAG pipeline that:

- Retrieves relevant information from internal documents
- Generates answers grounded strictly in retrieved content
- Avoids hallucinations and unsupported claims
- Clearly displays retrieved context and final answers
- Demonstrates understanding of RAG design choices and limitations

---
## 📁 Repository Structure
<pre>
mini-asgmt/
├── data/
│   ├── doc1.md
│   ├── doc2.md
│   └── doc3.md
│
├── rag.py                     # RAG using hosted LLM (OpenRouter)
├── rag_local_llm.py            # RAG using local open-source LLM (Ollama)
├── evaluation_questions.json   # 12 evaluation questions
├── evaluate_rag.py             # Evaluation & comparison script
└── README.md
</pre>

---

## 📄 Document Processing

### Chunking Strategy

Documents are chunked using **section-based chunking** based on Markdown headers (`##`).

**Why section-based chunking?**
- Preserves semantic coherence
- Avoids mixing unrelated topics
- Improves retrieval relevance
- Well-suited for policy and FAQ documents

Each section is treated as one retrievable chunk.

---

## 🧠 Embeddings

**Embedding Model Used:**  
`sentence-transformers/all-MiniLM-L6-v2`

**Why this model?**
- Lightweight and fast
- High-quality semantic representations
- Widely used in real-world RAG systems
- Works efficiently with FAISS for local vector search

---

## 🔎 Vector Search

**Vector Store:** FAISS (`IndexFlatL2`)

**Why FAISS?**
- Efficient local semantic search
- No dependency on managed services
- Ideal for small-to-medium document collections

For each user query, the system retrieves the **top-K (K=3)** most relevant document chunks using semantic similarity.

---

## ✨ Answer Generation (Grounded LLM Usage)

Retrieved chunks are passed to an LLM with **explicit grounding instructions**.

### Grounding Enforcement

Both pipelines instruct the LLM to:
- Use **only** the retrieved context
- Avoid external or prior knowledge
- Avoid hallucinations
- Return a fallback response if the answer is not present

Example instruction:
> *“Use ONLY the information provided in the context below. Do not introduce new facts or outside knowledge.”*

---

## 🤖 RAG Pipelines Implemented

### 1️⃣ Hosted LLM Pipeline (`rag.py`)

- **LLM:** Hosted via OpenRouter  
  Example model: `mistralai/mistral-7b-instruct`
- **Latency:** ~1–3 seconds per query
- **Behavior:**
  - Produces grounded summaries
  - Can explain implicitly described mechanisms
  - More expressive and user-friendly answers

---

### 2️⃣ Local Open-Source LLM Pipeline (`rag_local_llm.py`)

- **LLM:** `gemma:2b` via Ollama
- **Runs entirely locally**
- **Latency:** ~25–45 seconds per query (CPU)
- **Behavior:**
  - Extremely conservative
  - Very strong hallucination avoidance
  - Limited implicit reasoning ability

---

## 🧪 Evaluation Methodology

We evaluated both pipelines using **12 test questions** derived directly from the internal documents  
(see `evaluation_questions.json`).

Evaluation criteria:
- Retrieval relevance
- Groundedness
- Presence of hallucinations
- Completeness of answers
- Latency comparison

The script `evaluate_rag.py` runs both pipelines on the same questions and prints results side-by-side.

---

## 📊 Key Evaluation Observations

### Example Question  
**“What factors affect construction project delays?”**

#### Hosted LLM
- Correctly summarizes delay-related mechanisms
- Infers factors from documented processes
- Fully grounded in retrieved context

#### Local LLM
- Initially refuses to answer:
  > “The context does not provide any information…”
- Answers only when the question is reframed to match explicit document phrasing

---

## ⚠️ Why the Local Model Struggles

The local open-source model (`gemma:2b`) requires **explicit phrasing that closely matches the wording used in the documents**. It does not reliably infer causal relationships unless they are stated verbatim or framed in the same descriptive manner as the source text.

For example, when asked *“What factors affect construction project delays?”*, the model initially refused to answer because the documents describe **delay management mechanisms** rather than explicitly listing “factors.” However, when the prompt was reframed to align with the document language (e.g., *“What delay-related mechanisms or processes are described in the documents?”*), the local model successfully generated a grounded response.

This behavior demonstrates a **capability trade-off rather than a system failure**: smaller local models prioritize literal grounding and safety over implicit reasoning, whereas larger hosted models can perform controlled summarization and causal interpretation from the same retrieved context.


---

## 📈 Model Comparison Summary

| Aspect | Hosted LLM | Local LLM |
|------|----------|----------|
Groundedness | High | Very High |
Hallucinations | None | None |
Implicit reasoning | Yes | Limited |
Latency | Low | High |
Answer usefulness | Higher | Conservative |
Reliability | High | High |

---

## 🧠 Key Insight

> Smaller local LLMs prioritize safety and literal interpretation over implicit reasoning, while hosted models are better at grounded summarization when documents describe mechanisms rather than explicit answers.

This limitation is **intentionally documented** as part of the quality analysis.

---

## ▶️ How to Run Locally

### Environment Setup
```bash
conda create -n rag python=3.11
conda activate rag
pip install sentence-transformers faiss-cpu openai
```

### Hosted RAG
```bash
export OPENROUTER_API_KEY=your_key_here
python rag.py
```

### Local RAG
```bash
brew install ollama
brew services start ollama
ollama pull gemma:2b
python rag_local_llm.py

```
### Evaluation
```bash
python evaluate_rag.py
```

## ✅ Conclusion

This project demonstrates a complete, transparent, and grounded RAG system with:

- Structured document chunking
- Semantic retrieval using FAISS
- Strict grounding enforcement
- Transparent answer generation
- Real evaluation and model comparison

The comparison between hosted and local LLMs highlights practical trade-offs encountered in real-world RAG systems.

## 🏁 Final Status

- ✅ All mandatory requirements completed
- ✅ Local open-source LLM implemented
- ✅ Model comparison performed
- ✅ Evaluation and quality analysis documented

# 🩺 RAG-Based Medical Chatbot

A **Retrieval-Augmented Generation (RAG)** medical chatbot that answers over 100 types of healthcare queries using semantic search and large language models.

Built using **Flask**, **Pinecone**, and **Hugging Face Transformers**, the chatbot provides context-aware and accurate responses based on medical literature from the **Gale Encyclopedia of Medicine**.

---

## 🧠 Features

- 🔍 **Semantic Search**: Uses vector embeddings to find relevant medical knowledge chunks.
- 🧾 **Medical Corpus**: Embedded **1,000+ text chunks** from the Gale Encyclopedia of Medicine.
- 🗣️ **LLM Generation**: Integrated **Mistral-7B-Instruct-v0.1** (7B parameter model) for detailed answers.
- 🌐 **Web Interface**: Built with **Flask** for interactive medical Q&A.

---

## 💡 How It Works

1. **User submits a query** via the Flask frontend.
2. Query is **embedded** using the `e5-small-v2` model (384-dim).
3. **Pinecone** is used to retrieve top matching medical text chunks.
4. Retrieved context and user query are passed to the **Mistral-7B-Instruct-v0.1** model via `mistralai`.
5. The chatbot **generates a detailed, medically relevant response**.

---

## 🔧 Tech Stack

| Component     | Description                                      |
|---------------|--------------------------------------------------|
| 🧠 LLM         | Mistral-7B-Instruct-v0.1                         |
| 🔍 Embeddings  | e5-small-v2 (384-dimension)                     |
| 🧠 Vector DB   | Pinecone (for semantic search)                  |
| 🌐 Backend     | Flask                                            |
| 📚 Data Source | Gale Encyclopedia of Medicine (1,000+ chunks)   |

---

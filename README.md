# 🩺 Medical Chatbot Agent using LangChain + PubMed + FAISS

This project is an AI-powered medical chatbot built using **LangChain**, **Mistral-7B**, and **FAISS**, capable of answering medical questions by combining internal clinical documents with live PubMed search. It uses an **Agentic RAG** setup to dynamically choose between two tools:

- 🧠 **Internal Medical QA** – Answers using indexed clinical/medical PDFs via vector similarity search
- 🔬 **PubMed Search** – Queries real-time research papers using the PubMed API

---

## 🔧 Tech Stack

- **LLM**: Mistral-7B via Hugging Face Inference API
- **LangChain**: For chaining, agent orchestration, and memory
- **FAISS**: For fast vector similarity search on internal documents
- **PubMedQueryRun**: LangChain community tool for live research retrieval
- **HuggingFaceEmbeddings**: For vectorizing documents
- **Python**: Backend logic
- **.env**: For storing Hugging Face API token securely





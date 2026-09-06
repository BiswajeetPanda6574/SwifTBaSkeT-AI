# SwifTBaSkeT AI

### Hybrid SQL + RAG Business Intelligence Assistant

SwifTBaSkeT AI is a business intelligence assistant that combines structured SQL analytics with grounded Retrieval-Augmented Generation (RAG). The system uses hybrid query routing because vector search alone is not suitable for analytical questions such as "What are the top 5 areas by average order value?". Analytical queries are routed to SQL for deterministic computation, while contextual and record-level questions are handled through RAG. An evidence gate evaluates retrieved context and allows the system to abstain when evidence is insufficient, reducing the risk of unsupported or hallucinated answers.

## 🚀 Live Demo

**Streamlit:** https://swiftbasket-ai-ygchqgmmgzuf4ty6wg2gsg.streamlit.app/

**FastAPI API Docs:** https://swiftbasket-ai.onrender.com/docs

> **Note:** The FastAPI backend runs on Render's free tier and may spin down after inactivity. The first request after a period of inactivity can take approximately 30–60 seconds while the service wakes up.

## 🏗️ Architecture

```mermaid
flowchart LR
    U[User] --> S[Streamlit]
    S --> F[FastAPI - Render]
    F --> R[Query Router]

    R --> SQL[SQL Route]
    R --> RAG[RAG Route]

    SQL --> N[Neon PostgreSQL]
    N --> A[Answer]

    RAG --> E[FastEmbed / ONNX]
    E --> V[pgvector / HNSW]
    V --> G[Evidence Gate]

    G -->|Sufficient Evidence| GM[Gemini]
    G -->|Insufficient Evidence| AB[Abstain]

    GM --> A
    AB --> A

pip install -r render_requirements.txt

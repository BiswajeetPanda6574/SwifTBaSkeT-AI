# SwifTBaSkeT AI

### Hybrid SQL + RAG Business Intelligence Assistant

SwifTBaSkeT AI is a business intelligence assistant that combines structured SQL analytics with grounded Retrieval-Augmented Generation (RAG). The system uses **hybrid query routing** because vector search alone is not suitable for analytical questions such as "What are the top 5 areas by average order value?". Analytical queries are routed to SQL for deterministic computation, while contextual and record-level questions are handled through RAG. An evidence gate evaluates retrieved context and allows the system to **abstain when evidence is insufficient**, reducing the risk of unsupported or hallucinated answers.

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

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| FastAPI | Backend API |
| Streamlit | Frontend |
| PostgreSQL / Neon | Relational database |
| pgvector + HNSW | Vector search |
| FastEmbed / ONNX | Local embeddings |
| BAAI/bge-small-en-v1.5 | Embedding model |
| Google Gemini API | SQL generation & grounded responses |

**Deployment:** Streamlit Community Cloud + Render  
**Infrastructure:** Free-tier tools only — no OpenAI, LangChain, or paid infrastructure.

## 🧠 Key Engineering Decisions

- **Hybrid SQL + RAG:** SQL handles analytical queries and aggregations; RAG handles contextual and record-level queries.
- **Exact Order-ID routing:** Semantic search was unreliable for exact identifiers, so Order IDs are detected and retrieved directly from PostgreSQL.
- **Evidence gate:** Retrieved evidence is evaluated before generation, allowing the system to `ACCEPT`, `REVIEW`, or `ABSTAIN` when evidence is insufficient.
- **FastEmbed migration:** Replaced `sentence-transformers`/PyTorch with FastEmbed/ONNX to reduce memory usage and fit Render's 512 MB free-tier constraint.

## 📊 Example Queries

| Query | Route |
|---|---|
| What is the average order value? | SQL |
| What are the top 5 areas by average order value? | SQL |
| Show me the details and status of order `ORD0300000` | Exact-ID RAG |
| What payment methods are available? | Semantic RAG |
| Who won the 2023 Cricket World Cup? | Abstain |

## 💻 Local Setup

```bash
git clone https://github.com/BiswajeetPanda6574/SwifTBaSkeT-AI.git
cd SwifTBaSkeT-AI

pip install -r render_requirements.txt

# Jobot — Agentic RAG Career Assistant 

**Jobot** is an agentic, Retrieval-Augmented Generation (RAG) career assistant developed and deployed during my internship at **ESPRIT Tech**.  
This production-style deployment uses **LangChain / LangGraph** for multi-agent orchestration, **Milvus** as the vector database, **vLLM** for local LLM inference (GPU-accelerated), and **Chainlit** for an interactive text/voice UI. The system supports PDF/DOCX uploads, hybrid search + reranking, web search tools, and multi-agent workflows.

---

## Key features

- **Multi-agent orchestration** (LangChain + LangGraph) with specialized agents: CV analyzer, cover-letter generator, job matcher, web-search agent, orchestrator.
- **Retrieval (hybrid search)** combining sparse + dense retrieval and a reranker to improve relevance for long documents.
- **Vector DB:** Milvus for scalable vector storage and retrieval.
- **Local/accelerated inference:** vLLM + CUDA/RTX support for low-latency LLM calls (optional remote LLMs also supported).
- **Interactive UI:** Chainlit-based web app with text + voice I/O, PDF/DOCX upload, and session history.
- **Deployment-ready:** Docker Compose orchestration (GPU-aware) for multi-container deployment (API, worker, Milvus, vLLM).
- **Tooling:** Web search integration (for up-to-date query augmentation), PDF parsing, prompt/context engineering.

---

## Tech stack (short)

**Orchestration / RAG:** LangChain, LangGraph, hybrid search, reranker  
**Vector DB:** Milvus  
**LLM runtime:** vLLM (GPU), optional external LLM APIs  
**UI:** Chainlit  
**Backend:** Python
**Deployment:** Docker Compose, NVIDIA Container Toolkit (GPU)  

---

## Prerequisites (host machine)

- Docker & Docker Compose (v2) installed.  
- **If using GPU inference (recommended for vLLM):**
  - NVIDIA drivers + `nvidia-container-toolkit` .
  - CUDA-compatible GPU (RTX recommended).  
- (Optional) Python 3.10+ if you run components locally.

---

## Install & run (recommended: Docker Compose, GPU-enabled)

> **Important:** This deployment assumes the `docker/` compose file configures the Milvus service, vLLM worker, Chainlit service, and any helper workers. The commands below run everything inside containers so you don't need to manage Python envs locally.

Clone repository

git clone https://github.com/BettaiebHedi/Jobot-Agentic-RAG-Career-Advisor.git
cd jobot

Copy and configure environment file
cp .env.example .env
# edit .env to set MILVUS_HOST, MILVUS_PORT, MODEL_NAME, VLLM_MODEL_PATH, etc.
Start services with GPU support (recommended)

# if your docker CLI supports `docker compose`
docker compose -f docker/docker-compose.yml up --build -d
If your Docker requires nvidia runtime for GPU containers, ensure docker compose honors deploy / runtime options in the compose file and nvidia-container-toolkit is installed.

Initialize Milvus collections & embeddings (run the script once)

docker compose exec backend python milvus/milvus_setup.py
# or run it inside the backend container:
# docker compose exec backend python -m milvus.milvus_setup

Open Chainlit UI
The Chainlit UI should be exposed (e.g. http://localhost:8000 or the port specified in compose).

Use the web UI to upload CVs, chat with agents, and test job matching.

Stopping
docker compose down


Quick local dev (without Docker) — developer mode
If you prefer local dev (not recommended for production GPU inference):

Create virtualenv & install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Start Milvus locally (use Docker if needed) and ensure MILVUS_HOST env var points to it.

Start the Chainlit app
cd chainlit
chainlit run app.py -w


Hybrid search & reranker (how it works)
Sparse retrieval: (BM25 or simple keyword filter) quickly narrows candidate docs.
Dense retrieval: embed query + documents, query Milvus for nearest vectors.
Hybrid scoring: combine sparse and dense scores (weighted sum) to surface candidates.
Reranker: small cross-encoder or lightweight reranker re-scores top-K candidates to refine final context passed to LLM.
Context engineering: orchestrator merges top-ranked passages and crafts the prompt for the targeted agent.


License
This repository is released under the MIT License. See LICENSE.

Author
Mohamed el hedi Bettaieb — AI Engineer
Siwar Jlassi — AI Engineer
Sirine Nmiri — AI Engineer

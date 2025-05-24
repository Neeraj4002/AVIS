```text 
=============================================
    ___ _    ___________           ____   ___
   /   | |  / /  _/ ___/   _   __ / __ \ <  /
  / /| | | / // / \__ \   | | / // / / / / / 
 / ___ | |/ // / ___/ /   | |/ // /_/ / / /  
/_/  |_|___/___//____/    |___(_)____(_)_/   
                                             
=============================================

```
# 🚀 Avis: Agentic Code Generation System

Avis is an **LLM-based code generation agent** purpose-built to deliver **accurate, production-ready code** using a Retrieval-Augmented Generation (RAG) pipeline over framework documentation.

> 📌 Mission: To eliminate hallucination and boost developer productivity by grounding every response in real, version-aware documentation.

---

## 📦 What It Does

Avis is designed to:
- Ingest and embed **framework documentation** (e.g., LangGraph, FastAPI, PyTorch).
- Retrieve highly relevant chunks using **hybrid retrieval** and **custom rerankers**.
- Generate precise code using **instruction-following LLMs** (OpenAI, Gemini, Claude, etc.).
- Automatically **self-evaluate, rerun, and correct** outputs to ensure consistent reliability.

---

## 🧠 System Overview
```
User Prompt
│
▼
[Input Parser] → [Retriever (Semantic + Filtered)] → [Reranker]
│ │
└────────────────────→ [Code Generation Agent] ←───┘
│
▼
[Self-Checker + Rewriter Loop]
│
▼
Final Code Output ✅
```

---

## ⚙️ Key Features

- 📚 **Framework-Aware RAG:**  
  Parses and chunks docs using custom strategies, retaining structure (functions, classes, examples).

- 🔎 **Hybrid Retrieval:**  
  FAISS + keyword filters + metadata (e.g., version, module, type).

- 🧠 **Agentic Code Loop:**  
  - `observe`: Analyze prompt & retrieve context  
  - `think`: Choose generation strategy (e.g., template, scratch, edit)  
  - `act`: Generate code  
  - `evaluate`: Check syntax/errors/tests  
  - `revise`: Rerun or correct if needed  

- 📈 **Accuracy-First Design:**  
  - Zero-shot eval  
  - AST comparisons  
  - Task-level metrics (pass@1, correctness, hallucination rate)

- 🔌 **Plug & Play Support for LLMs:**  
  OpenAI GPT-4, Claude Opus, Gemini 1.5, Mistral, etc.

---

## 🔧 Setup

```bash
# Clone the repo
git clone https://github.com/your-org/avis.git && cd avis
```
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
# (Optional) Install FAISS
pip install faiss-cpu
```

#📁 Project Structure
```
avis/
├── indexer/            # Chunking + embedding from docs
├── retriever/          # Vector + filter-based retrieval
├── agent/              # LLM orchestration + evaluation loop
├── server/             # FastAPI wrapper for web-based usage
├── examples/           # Prompt → Code → Eval demos
├── configs/            # YAML configs for pipelines
└── tests/              # Eval + test cases
```
#🚀 Usage
1. Embed Documentation
```
python -m avis.indexer --config configs/langgraph.yml --docs ./docs/langgraph/
```
2. Run Server
```
uvicorn server.main:app --reload
```
3. Query Agent (via API or CLI)
```
curl -X POST http://localhost:8000/code \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a LangGraph node that retries if error occurs"}'
```
✅ Sample Output
Prompt:

Create a LangGraph node that retries a step on error with exponential backoff.

Output:
```python
from langgraph.graph import StateGraph
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def risky_node(state):
    # Your logic here
    return state

builder = StateGraph()
builder.add_node("retry_node", risky_node)
```
✅ Backed by docs: langgraph/StateGraph, tenacity.retry, etc.
✅ Passed syntax + lint check

📌 Roadmap
 GitHub Copilot integration

 Multi-version doc support

 Auto-patch hallucinated code

 VSCode extension


📄 License
MIT License. See LICENSE for details.

Made with ⚡ by [S.N.K] — AI for Devs, Done Right.









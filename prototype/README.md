# ECO-EDGE research prototype (LangGraph)

Functional prototype of the ActSimSecCrit pipeline: each phase (intent parsing, planning,
digital-twin simulation, security check, critic, execution) is a node in a LangGraph cyclic
state graph; SLMs are served locally with Ollama (default: phi3:mini).

Requirements: `pip install -r requirements.txt`, plus a running Ollama with `ollama pull phi3:mini`.
Run from this directory so that `agents/`, `graph/`, `orchestration/` and `simulation/` are importable,
or use the Dockerfile / docker-compose.yml.

This prototype is used for qualitative validation only; all quantitative results in the paper come
from `../ecoedge_sim.py`.

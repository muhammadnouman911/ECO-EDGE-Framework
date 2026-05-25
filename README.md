# ECO-EDGE: Decentralized Agentic Orchestration Framework

This repository contains the simulation code, prototype implementation, and baseline comparisons for the paper:
**"ECO-EDGE: A Decentralized Agentic Orchestration Framework for Energy-Efficient Edge Computing Using Small Language Models"**

## 📖 Overview
ECO-EDGE is a novel agentic architecture designed for sustainable 6G and Multi-access Edge Computing (MEC) environments. By deploying Small Language Models (SLMs) as active "thinking units" locally on edge devices, ECO-EDGE balances orchestration intelligence with critical energy constraints.

### Key Innovations:
1. **ActSimSecCrit Pipeline:** A 4-phase validation loop ensuring intelligent, physically feasible, and prompt-injection-secure network actuations.
2. **Elastic Intelligence:** A dynamically scaling tier system that adjusts SLM inference complexity based on real-time hardware battery limits.
3. **Agentic Efficiency ($\eta_A$):** A formal mathematical metric proving the net positive gain of running inference vs. traditional blind execution.
4. **Cross-Lingual Intent:** Multi-lingual intent grounding mapping NLP to JSON APIs.

## 🗂️ Repository Structure

* `baselines/`: Heuristic implementations of state-of-the-art frameworks (CORE, ARC, ORION) for objective comparison.
* `config/`: System and hardware configuration files.
* `core/`:
  * `agentic_engine.py`: Contains the `ActSimSecCrit` LangGraph compilation logic.
  * `edge_environment.py`: Simulated digital twin representing heterogeneous hardware (Raspberry Pi, Jetson, etc.) and their constraints.
  * `prompts.py`: SLM persona configurations and safety constraints.
* `experiments/`: Scripts used to generate empirical datasets and graphs.
  * `run_ablation.py`: Executes tests for "No Critic" vs. "ActSimCrit".
* `utils/`: Networking stubs and math telemetry plotters.
* `main_simulation.py`: Entry point for executing a full 1000-device simulated run.

## 🚀 How to Run the Simulations

### Prerequisites
* Python 3.9+
* `pip install -r requirements.txt` (Main dependencies: `langgraph`, `langchain`, `matplotlib`, `numpy`, `simpy`)
* Optional (For Real SLM Execution): Install [Ollama](https://ollama.ai/) and pull models: `ollama run phi3:mini` or `ollama run tinyllama`.

### Executing Benchmark Comparisons (Large-Scale)
To reproduce the numerical results shown in the paper's figures (Success Rate, Latency, Energy):

```bash
python experiments/run_ablation.py
```
*Note: This script uses a mathematically bounded probabilistic digital twin to simulate scaling up to 1000 devices without needing a massive physical cluster.*

### Executing Local LangGraph Prototype (Tier-2 Node)
To test the real semantic capabilities of the Local Orchestrator using an active SLM:

```bash
export USE_SLM=true
export OLLAMA_MODEL=phi3:mini
python main_simulation.py
```
*This will execute the multi-step `ActSimSecCrit` graph and measure the prototype's end-to-end local latency.*

## 📊 Result Interpretation

The `experiments/` outputs generate standard datasets which plot exactly to the following claims found in Section VI of the paper:
- **Decision Success:** Up to 3.6x improvement over generalized heuristics like Next-Fit.
- **Energy Overhead:** Curves plateau aggressively post-700 devices indicating active `Elastic Intelligence`.
- **Latency:** Maintains sub-second latency via distributed intent parsing unlike centralized Cloud LLM controllers.

## 📜 Citation
If you use this code or framework in your research, please cite our IEEE paper:
*(Citation details will be updated post-publication)*

# ECO-EDGE: Decentralized Agentic Orchestration Framework

> **Research prototype** accompanying the paper:  
> *ECO-EDGE: A Decentralized Agentic Orchestration Framework for Energy-Efficient Edge Computing Using Small Language Models*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

---

## Overview

ECO-EDGE is a **multi-agent orchestration system** for 6G Multi-access Edge Computing (MEC) environments. It uses **Small Language Model (SLM) "Thinking Units"** deployed across a hierarchical edge-cloud continuum to perform energy-efficient, intent-driven resource management.

This repository provides a **runnable LangGraph prototype** that demonstrates the core **ActSimSecCrit** decision pipeline.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Tier 3: Cloud                           │
│   ┌──────────────────┐    ┌─────────────────────┐       │
│   │ Global State DB  │───▶│ Core LLM (70B+)     │       │
│   └──────────────────┘    │ Policy Evolution     │       │
│                           └─────────────────────┘       │
└────────────────────────┬─────────────────────────────────┘
                         │ Intent & State Sync
┌────────────────────────▼─────────────────────────────────┐
│              Tier 2: MEC Server                          │
│   ┌──────────────────┐    ┌─────────────────────┐       │
│   │ Regional Planner │◄──▶│ Planner SLM (3B)    │       │
│   │ Role-Affinity    │    │ ActSimSecCrit        │       │
│   └──────────────────┘    └────────┬────────────┘       │
│                                    │                     │
│                           ┌────────▼────────────┐       │
│                           │ Digital Twin         │       │
│                           │ Simulation Engine    │       │
│                           └─────────────────────┘       │
└──────────┬───────────────────────────────┬───────────────┘
           │ RPC                           │ RPC
┌──────────▼──────────┐         ┌──────────▼──────────┐
│ Tier 1: Edge Node A │         │ Tier 1: Edge Node B │
│ ┌─────────────────┐ │         │ ┌─────────────────┐ │
│ │ Elastic SLM     │ │◄───────▶│ │ Execution SLM   │ │
│ │ (0.5B-1.1B)     │ │ A2A Sync│ │ (1.1B)          │ │
│ ├─────────────────┤ │         │ ├─────────────────┤ │
│ │ Battery Monitor │ │         │ │ Telemetry       │ │
│ ├─────────────────┤ │         │ ├─────────────────┤ │
│ │ Actuators       │ │         │ │ Actuators       │ │
│ └─────────────────┘ │         │ └─────────────────┘ │
└─────────────────────┘         └─────────────────────┘
```

---

## Agent Pipeline: ActSimSecCrit

The framework validates every orchestration decision through a **multi-agent pipeline**:

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
Intent ──▶ Planner ──▶ Simulate ──▶ Security ──▶ Critic ──▶ Execute
              ▲                                    │
              └──────── Re-plan (max 3) ◄──────────┘
                         (if rejected)
```

| Phase        | Agent             | Role                                                  |
|-------------|-------------------|-------------------------------------------------------|
| **Intent**  | `IntentAgent`     | Parse NL intent → structured goals & constraints      |
| **Act**     | `PlannerAgent`    | Generate candidate actions (sleep, migrate, steer)    |
| **Simulate**| `SimulationAgent` | Predict KPIs via digital twin                         |
| **Security**| `SecurityAgent`   | Role auth + prompt integrity + anomaly detection      |
| **Critic**  | `CriticAgent`     | Threshold evaluation → approve / reject               |
| **Execute** | `ExecutionAgent`  | Apply validated action to the environment             |

---

## Project Structure

```
eco-edge-framework/
├── agents/
│   ├── intent_agent.py          # NL intent parsing
│   ├── planner_agent.py         # Action generation
│   ├── simulation_agent.py      # Digital twin simulation
│   ├── security_agent.py        # 3-phase security checks
│   ├── critic_agent.py          # Threshold-based evaluation
│   └── execution_agent.py       # Action execution
├── orchestration/
│   ├── role_affinity_scheduler.py   # Task-to-agent matching
│   └── elastic_intelligence.py      # Battery-aware MDP scaling
├── graph/
│   └── eco_edge_graph.py        # LangGraph StateGraph workflow
├── simulation/
│   └── edge_environment.py      # 3-tier MEC network simulator
├── experiments/
│   └── run_experiment.py        # CLI experiment runner
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd eco-edge-framework
pip install -r requirements.txt
```

### 2. Run the Default Experiment

```bash
python experiments/run_experiment.py
```

### 3. Custom Intent

```bash
python experiments/run_experiment.py \
  --intent "Reduce energy consumption by 30% while keeping latency under 40ms" \
  --nodes 50 \
  --seed 123
```

### 5. Docker Support (Recommended for Portability)

To ensure a consistent environment for reviewers, you can run the framework using Docker:

```bash
# Build the image
docker build -t eco-edge .

# Run the default experiment
docker run eco-edge
```

Alternatively, use **Docker Compose**:

```bash
docker-compose up
```

### 6. Small Language Model (SLM) Integration (Optional)

ECO-EDGE supports real-time reasoning using local SLMs via [Ollama](https://ollama.com/).

**Setup:**
1. Install Ollama from [ollama.com](https://ollama.com/).
2. Pull a compact model: `ollama pull phi3:mini`.
3. Start the Ollama server (usually runs on `http://localhost:11434`).
4. Set environment variables to enable SLM:
   ```bash
   $env:USE_SLM="true"
   $env:OLLAMA_MODEL="phi3:mini"
   python experiments/run_experiment.py
   ```

When `USE_SLM` is `true`, the Intent, Planner, and Critic agents will use the local model for advanced reasoning. If the model is unavailable, they will automatically fall back to the rule-based logic.

---

## Example Scenario

**Input Intent:**
> "Reduce energy consumption by 25% in the edge network while keeping latency under 50 ms."

**Pipeline Execution:**

1. **IntentAgent** extracts: energy reduction goal (25%), latency constraint (≤50ms)
2. **PlannerAgent** proposes: composite action (micro-sleep 6 nodes + migrate workload to MEC + steer traffic)
3. **SimulationAgent** predicts: latency=32ms, energy reduction=28%, success rate=92%
4. **SecurityAgent** validates: role authorization ✓, prompt integrity ✓, anomaly detection ✓
5. **CriticAgent** evaluates: all thresholds met → **APPROVED ✓**
6. **ExecutionAgent** applies: 8 changes to the network, new power consumption reduced

---

## Key Innovations

- **Elastic Intelligence**: Battery-aware MDP that dynamically scales SLM complexity (2.7B → 1.1B → 0.5B) based on device charge level
- **ActSimSecCrit Pipeline**: Four-phase validation ensuring no unverified action reaches the live network  
- **Role-Affinity Scheduling**: Semantic matching of tasks to specialized agents
- **Re-planning Loop**: Automatic retry with conservative adjustments on critic rejection

---

## Configuration

| Flag             | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `--intent`      | (example) | Natural language orchestration intent |
| `--nodes`       | 20      | Number of Tier-1 edge nodes          |
| `--mec`         | 5       | Number of Tier-2 MEC servers         |
| `--cloud`       | 2       | Number of Tier-3 cloud nodes         |
| `--max-attempts`| 3       | Maximum re-planning attempts         |
| `--seed`        | 42      | Random seed for reproducibility      |
| `--json`        | false   | Output raw JSON state                |

---

## Future Research Extensions

- [ ] Connect agents to real LLM backends (OpenAI, Ollama, vLLM)
- [ ] Add federated learning for distributed model updates
- [ ] Integrate with EdgeCloudSim for large-scale Java-based simulation
- [ ] Hardware-in-the-loop validation on Jetson / Raspberry Pi
- [ ] Multi-language intent support (Cross-Lingual Intent Mapping)
- [ ] Adversarial robustness testing for the Security-Critic

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{nouman2025ecoedge,
  title={ECO-EDGE: A Decentralized Agentic Orchestration Framework for 
         Energy-Efficient Edge Computing Using Small Language Models},
  author={Nouman, Muhammad and Khan, Shanzay and Awan, Kamran Ahmed},
  year={2026},
  institution={University of Haripur}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

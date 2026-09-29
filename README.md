# ECO-EDGE: Decentralized SLM-based Agentic Orchestration for Edge Computing

Code, trained policy, raw results and figure scripts for the paper
**ECO-EDGE: A Decentralized Small-Language-Model Agentic Orchestration Framework for Edge Computing, with a Quantified Inference Cost**
(M. Nouman, S. Khan, M. Ibrahim, C. Choi).

Every number, table and figure in the paper is produced from this repository.

## Repository layout

```
ecoedge_sim.py        mechanistic discrete-event simulator + five orchestrators
                      (ECO_EDGE, NEXT_FIT, RANDOM_FIT, DQN_FIT = tabular Q-learning, PPO_FIT)
ppo_train.py          NumPy PPO trainer (paper hyper-parameters)
ppo_weights.npz       trained PPO policy (1,000,000 decisions at N=200)
ppo_weights_log.json  PPO training curve;  ppo_train.log = console log
run_experiments.py    scale sweep, ablation and sensitivity sweeps (resumable)
analyze.py            aggregates results/*.jsonl -> analysis/ (tables, summary.json, figures)
results/              raw results, one JSON record per run
                        scale.jsonl        2,500 runs (100-1000 devices x 50 seeds x 5 orchestrators)
                        ablation.jsonl       450 runs (9 configurations x 50 seeds, N=1000)
                        sensitivity.jsonl  1,060 runs (arrival rate, bandwidth, security FPR,
                                                       SoC thresholds; 20 seeds, N=1000)
analysis/             generated LaTeX tables, summary.json and figures/ used in the paper
experiments/          Security-Critic evaluation (55 adversarial + 30 benign prompts)
                        evaluate_adversarial_55.py --mode rules  -> adversarial_results_55.json,
                                                                    adversarial_table_latex.txt
                        evaluate_cross_lingual.py  qualitative cross-lingual grounding demo (needs Ollama)
prototype/            LangGraph/LangChain research prototype of the ActSimSecCrit pipeline
                      (agents/, graph/, orchestration/, simulation/; needs Ollama + Phi-3:mini)
figures/              architecture and ActSimSecCrit diagrams
```

## Requirements

* Simulation, training and analysis: Python 3.10+, `numpy`, `scipy`, `matplotlib`.
  No PyTorch and no Java are required.
* Prototype only: see `prototype/requirements.txt` (LangGraph, LangChain, Ollama).

```
pip install numpy scipy matplotlib
```

## Reproducing the paper

```
python ppo_train.py --steps 1000000 --n-edge 200 --out ppo_weights.npz   # ~5 min on one core
python run_experiments.py --exp all --seeds 50                           # ~2 h on one core; resumable
python analyze.py --results results --out analysis
python experiments/evaluate_adversarial_55.py --mode rules
```

Seeds are fixed (1..50 for the main comparison and ablation, 1..20 for sensitivity);
every run is deterministic given its seed, and `run_experiments.py` skips runs that are
already present in `results/`, so an interrupted sweep can simply be restarted.

`analyze.py` prints the N=1000 summary, the Welch t-tests with Cohen's d, and the ablation
table, and writes `analysis/table_*.tex`, `analysis/summary.json` and `analysis/figures/*`.

## What the simulator models

* Edge devices with heterogeneous compute, link bandwidth, power, accelerator specialisation
  and (for 40% of devices) a battery budget; a fixed set of MEC servers with a shared uplink
  access point each.
* Three task classes (traffic steering, security-critical, routine sensing) with Poisson
  arrivals, per-class data/compute/deadline ranges, and 2% adversarial resource-exhaustion
  requests.
* FIFO queues with capacity limits, transfer delays with access-point contention,
  accelerator-dependent processing times, deadline checking, battery drain and node depletion.
* Orchestrator decision latency and energy: SLM tiers (0.5B/1.1B/2.7B: 22/45/120 ms,
  0.15/0.31/0.84 J, grounding accuracy 0.65/0.82/0.95), heuristics (1 ms, 1 mJ), RL (2 ms, 5 mJ).
* ECO-EDGE components can be switched off individually (`Config` flags) for the ablation.

No orchestrator is given a success probability; all outcomes emerge from the mechanics above.

## Security-Critic measurement

`experiments/evaluate_adversarial_55.py --mode rules` runs the rule-based prompt-integrity
layer on 55 adversarial prompts (six attack families) and 30 benign prompts. Result:
44/55 detected (80.0%), 0/30 false positives. These two rates are the simulator's
`sec_detect` / `sec_fpr` parameters; the false-positive rate is additionally swept 0-20%
in the sensitivity experiment.

## Prototype

`prototype/` contains the LangGraph implementation of the Act -> Simulate -> Security ->
Critic pipeline with Ollama-served Phi-3:mini, used for qualitative validation of the
pipeline logic and for the cross-lingual grounding demo. It is not used to produce any
quantitative result in the paper. Run from inside `prototype/` (see its Dockerfile).

## License

See `LICENSE`.

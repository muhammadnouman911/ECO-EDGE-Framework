# Files to add to github.com/muhammadnouman911/ECO-EDGE-Framework

Copy these into the repository at the SAME paths and commit:

  mdp_vi.py                 (new)  value iteration for the tier-selection MDP (paper Eq. 11, Section IV-C)
  run_vi_ablation.py        (new)  runs ECO_EDGE with VI-optimal thresholds, 50 seeds, N=1000
  analyze.py                (replace)  identical to the current file except 2 lines: "H_vi_optimal" added to `order` and `ALAB`
  results/ablation.jsonl    (replace)  current file + 50 appended rows labelled H_vi_optimal
  analysis/vi.json          (new)  value-iteration output (thresholds, value gaps)

Suggested commit message:
  Add value-iteration solution of the tier-selection MDP and VI-optimal ablation (OJ-COMS revision)

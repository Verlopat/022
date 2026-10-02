# GATC: camera-only coordinated signal control for heterogeneous traffic

Code, trained weights and raw results behind the manuscript
"Camera-Only Coordinated Signal Control for Heterogeneous Traffic: Graph Attention Estimation
of Hidden Queues with an Auditable Two-Level Timing Law".

Everything runs on a CPU with Python 3.11, NumPy, SciPy, pandas and matplotlib. No GPU,
PyTorch or SUMO is needed: the simulator and the automatic differentiation are included.

## Files

| File | Purpose |
|---|---|
| `simcore.py` | Mesoscopic corridor simulator (3 junctions x 4 arms x 2 lanes, 7 vehicle groups, spillback, split phasing), scenarios S1-S5 and S2HV, camera snapshot and detector error model (profiles P0-P3), Webster plan |
| `controllers.py` | FT-W, ACT, MP-C, MP-A, LPS (+ tactical), feature builder, behaviour policy, target computation and the proposed GATC controller (strategic layer + tactical camera layer) |
| `nn.py` | Reverse-mode autodiff on NumPy, dense multi-head GAT, GCN and MLP estimators, Adam; `python3 nn.py` runs a gradient check |
| `verify_webster.py` | Simulator verification against Webster's delay formula (Figure 6) |
| `datagen.py` | 400 training episodes under the exploratory behaviour policy (`python3 datagen.py 400`) |
| `train.py` | Estimator training (`python3 train.py gat:full gat:local gcn:full mlp:none`) |
| `train_seeds.py` | Two extra training seeds per estimator and the mean/SD summary |
| `tune.py` | Controller tuning on seeds 500-505 (all grids used in the paper) |
| `experiments.py` | Parallel batch runner and controller factory |
| `run_all.py` | Test phase on seeds 1-20: `main`, `ablation`, `series`, `perception`, `sweep` |
| `run_estctrl.py` | Control outcome for every estimator and training seed |
| `stats.py` | Wilcoxon signed-rank, Holm correction, rank-biserial effect size, bootstrap CIs |
| `tables_static.py`, `tables_results.py` | All manuscript tables |
| `fig_arch.py`, `figs_static.py`, `figs_results.py`, `figstyle.py` | All figures |
| `build.py` | Assembles the manuscript and writes the Word files |
| `rl.py` | DQN-C / DQN-O baselines: Q-network with manual backpropagation, double Q-learning, camera or exact-state observations |
| `run_rl.sh` | Launches one DQN training run (`./run_rl.sh <delta> <episodes> <seed> <tag> [key=value ...]`) |
| `run_dqn.py` | DQN checkpoint selection on tuning seeds 500-505, test runs on seeds 1-20, S3-S5 time series |
| `run_dqn_curve.py` | Tuning-seed evaluation of early checkpoints for the learning-curve figure |
| `audit_docx.py`, `check_render.sh`, `eqcheck.js` | Formatting audit of the Word file, page rendering and equation check |
| `results/final_configs.json` | Controller settings frozen after tuning |
| `results/*.jsonl` | Raw per-run results (one JSON object per run) |
| `models/*.pkl` | Trained estimator weights (seed 0 plus seeds 1 and 2) |

## Reproducing the paper

```
python3 verify_webster.py
python3 datagen.py 400
python3 train.py gat:full gat:local gcn:full mlp:none
python3 tune.py FTWD,FTWO,ACT5,MP-C,MPA2,LPS,LPST3,NGS,NGT2 2
python3 run_all.py main && python3 run_all.py ablation && python3 run_all.py series
python3 run_all.py perception && python3 run_all.py sweep
python3 train_seeds.py train gat:full:1,gat:local:1,gcn:full:1,mlp:none:1
python3 train_seeds.py train gat:full:2,gat:local:2,gcn:full:2,mlp:none:2
python3 train_seeds.py summary && python3 run_estctrl.py run && python3 run_estctrl.py summary
python3 tables_static.py && python3 tables_results.py
python3 fig_arch.py && python3 figs_static.py && python3 figs_results.py
# deep Q-network baselines (selected configurations; see Table 5 for the full search)
./run_rl.sh 10 300 0 dqn_d10_s0 alpha=0 lr=0.0005 obs_v1=1 backlog=0   # DQN-C (basic 46-feature layout)
./run_rl.sh 10 400 0 dqno_d10_g8 alpha=0 gmin=8 oracle=1  # DQN-O (exact state)
python3 run_dqn.py test dqn_d10_s0 225 DQN-C P1 v1 && python3 run_dqn.py test dqno_d10_g8 200 DQN-O P1
python3 run_dqn.py test dqn_d10_s0 225 DQN-C P0,P2,P3 v1
python3 run_dqn.py series dqn_d10_s0:225:DQN-C:v1,dqno_d10_g8:200:DQN-O
python3 build.py
```

Note on DQN-C: the selected configuration was trained with the first version of `rl.py`
(46 camera features, junction-queue reward without entry backlog, Adam 5e-4). The flags `obs_v1=1
backlog=0` reproduce that configuration with the current file, and `models/dqn_v1/` keeps the
original selection logs.

A single simulation run takes about 0.2-0.4 s; the full test phase takes about 15 min on two cores.
The detector profiles P1-P3 are modelling assumptions (see the paper, Table 4); replace them with
error rates measured on UVH-26 frames processed by the deployed detector before drawing field conclusions.
# 022

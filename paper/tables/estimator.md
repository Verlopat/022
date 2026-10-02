CAPTION: Table 7. Accuracy of next-cycle demand estimators on 80 held-out episodes (5,349 graph snapshots) and the resulting average waiting time (mean over S1-S5 and 20 test seeds) when each estimator drives GATC-S or GATC; learned estimators are mean ± SD over three training seeds.

| **Estimator** | **Edges used** | **MAE Q+A (PCU)** | **R² Q+A** | **R² internal arms** | **MAE Q (PCU)** | **MAE A (PCU)** | **R² ψ** | **GATC-S waiting (s/veh)** | **GATC waiting (s/veh)** |
|---|---|---|---|---|---|---|---|---|---|
| Raw snapshot score | None | 22.96 | -0.203 | -0.991 | - | - | - | - | - |
| Calibrated snapshot | None | 12.92 | 0.609 | 0.446 | - | - | -2.087 | - | - |
| MLP | Self-loops only | 8.62 ± 0.07 | 0.798 ± 0.000 | 0.636 ± 0.003 | 5.24 ± 0.07 | 5.78 ± 0.02 | 0.520 ± 0.001 | 130.2 ± 1.4 | 91.4 ± 0.1 |
| GAT-local | Intra-junction and hub | 8.23 ± 0.01 | 0.810 ± 0.002 | 0.670 ± 0.002 | 5.56 ± 0.02 | 5.42 ± 0.01 | 0.543 ± 0.003 | 138.3 ± 2.0 | 91.3 ± 0.5 |
| GCN | All four sets | 8.21 ± 0.05 | 0.814 ± 0.001 | 0.697 ± 0.001 | 5.79 ± 0.04 | 5.33 ± 0.02 | 0.603 ± 0.001 | 134.2 ± 1.0 | 90.9 ± 0.3 |
| GAT (proposed) | All four sets | 7.58 ± 0.06 | 0.839 ± 0.002 | 0.726 ± 0.006 | 5.14 ± 0.12 | 5.13 ± 0.02 | 0.656 ± 0.001 | 135.2 ± 2.6 | 91.5 ± 0.2 |

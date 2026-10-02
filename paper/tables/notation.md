CAPTION: Table 2. Notation.

| **Symbol** | **Meaning** |
|---|---|
| $j$, $i$ | Junction index (J1-J3) and arm index (W, N, E, S) |
| $I_g$, $g_{\min}$ | Intergreen (5 s) and minimum green (8 s) |
| $C_{j,k}$, $g_{ij,k}$ | Cycle length and green time of arm $i$ in cycle $k$ |
| $L$, $S$ | Lost time per cycle (24 s) and design saturation flow of a two-lane arm (1 PCU/s) |
| $D$, $r^P_c$, $o^P$, $\lambda^P$ | Camera range (60 m), recall, occlusion factor and false-positive rate of profile $P$ |
| $w_c$, $s_{ij}$ | PCU weight of group $c$ and PCU lane score |
| $G=(V,E)$, $\mathcal{N}(p)$ | Junction graph with 15 nodes and neighbourhood of node $p$ |
| $\alpha_{pq}$ | Attention coefficient from node $p$ to node $q$ |
| $\hat{Q}_{ij}$, $\hat{A}_{ij}$ | Predicted standing queue and arrivals in the next $H_t=120$ s (PCU) |
| $\hat{\psi}_j$ | Predicted junction flow ratio (impact score) |
| $\mu$, $\beta$ | Maximum-cycle scaling and cycle smoothing factor |
| $\kappa$, $\rho_0$, $\delta_{ij}$ | Guard strength, guard threshold and guard factor |
| $Q^{st}_m$, $\kappa_s$ | Storage of arm $m$ (car-spaces) and PCU per car-space (1.33) |
| $Z_{ij}(t)$ | Vehicles detected in the 20 m stop-line zone at time $t$ |
| $\gamma$, $\tau_k$, $r_k$, $Q_n$ | DQN-C discount per second, transition duration, reward and queue normaliser (40 PCU) |
| $\bar{W}$, $J$ | Average waiting time per vehicle and Jain fairness index |

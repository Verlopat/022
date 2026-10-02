CAPTION: Table 6. Controllers, sensing regimes and settings frozen after tuning on seeds disjoint from the test seeds.

| **Controller** | **Sensing** | **Search space (seeds 500-505)** | **Frozen setting** |
|---|---|---|---|
| FT-W | None | $C_{\max}\in\{160,200,240\}$ s | Design-day Webster plan; peak cycle 140 s, off-peak 70 s |
| FT-W* | None (true scenario flows) | $C_{\max}\in\{160,200,240\}$ s | $C_{\max}=200$ s |
| MP-C | Own and receiving arms, once per junction cycle | $C\in\{80,100,120,150\}$ s | $C=120$ s |
| LPS | Own 4 arms, once per junction cycle | $C\in\{90,120,150\}$ s | $C=120$ s |
| GATC-S | All 12 arms, once per junction cycle | $\kappa$, $\rho_0$, $\mu$, $C_{\max}$: 72 settings | $\kappa=0.5$, $\rho_0=0.5$, $\mu=1.0$, $C_{\max}=240$ s |
| ACT | Stop-line loops, every lane, continuous | max green 60-120 s, gap 2-3 s | max green 90 s, gap 2.0 s |
| MP-A | Own and receiving arms every $\Delta$ s | $\Delta\in\{10,15,20,30\}$ s, min green 8-30 s | $\Delta=20$ s, min green 20 s |
| LPS-T | As LPS, plus 1 Hz green-arm check | $C\in\{200,240,300\}$ s | $C=240$ s |
| DQN-C | Own and downstream arms every $\Delta$ s | Nine configurations (Table 5) × checkpoints | Basic features, junction-queue reward, $\Delta=10$ s, minimum green 8 s, episode 225 of 300 |
| GATC | As GATC-S, plus 1 Hz green-arm check | $\kappa$, $\mu$, $C_{\max}$: 24 settings | $\kappa=0.5$, $\rho_0=0.7$, $\mu=2.0$, $C_{\max}=240$ s |
| DQN-O (reference) | Exact simulator state every $\Delta$ s | Checkpoints of one configuration | Queue reward with entry backlog, $\Delta=10$ s, minimum green 8 s, episode 200 of 400 |

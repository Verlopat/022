CAPTION: Table 5. Configuration and training protocol of the estimators and of the DQN baselines.

| **Item** | **Setting** |
|---|---|
| Layers and width | 2 message-passing layers, 4 heads x 16 units (GAT); 64 units (GCN, MLP); ELU |
| Output heads | Arm: softplus for $(\hat{Q},\hat{A})$ scaled by $Y_s=40$ PCU; junction: sigmoid for $\hat{\psi}$ |
| Attention | LeakyReLU slope 0.2, masked softmax over $\mathcal{N}(p)$ with self-loops |
| Parameters | GAT 5,613; GCN 5,357; MLP 5,357 |
| Loss | Huber ($\delta=0.5$) on normalised $Q$ and $A$; squared error on $\psi$ with weight 2 |
| Gradient method | Adam ($\beta_1=0.9$, $\beta_2=0.999$), weight decay $10^{-4}$, batch 128 |
| Learning rate | Cosine schedule from $3\times10^{-3}$ to $3\times10^{-4}$ |
| Arm-node features (14) | PCU lane score; counts of four class groups; fill ratio of the camera zone; green share and length of the last cycle; score change since the previous frame; internal-arm and corridor-arm flags; green-now flag; time since the last green; node-type flag |
| Junction-node features | Mean, maximum and spread of arm scores; total detections; mean fill ratio; cycle length; mean score change; middle-junction flag; elapsed cycle fraction; node-type flag (other entries zero) |
| Behaviour policy | Cycle 60-160 s; greens from camera scores mixed with Dirichlet noise (weight 0-0.6); random or descending phase order; tactical layer in 60% of episodes |
| Training distribution (estimators and DQN-C) | Demand factor 0.5-1.35; eastbound and westbound entry ratios 0.6-1.6 and 0.6-1.4; lane blockage in 30% of episodes (random arm and lane, 600-1,500 s); heavy-vehicle mix in 20%; profiles P0/P1/P2/P3 with probabilities 0.10/0.60/0.15/0.15 |
| Epochs | 100 maximum, early stopping after 20 epochs without a lower held-out loss |
| Data | 400 episodes, 26,745 graph snapshots, episode-level split 80/20 |
| Training time (CPU, one core) | GAT 9.9 min; GCN 2.8 min; MLP 2.5 min; DQN-C 6.3 min; DQN-O 7.2 min |
| DQN-C and DQN-O network | Two hidden layers of 128 ReLU units, 4 outputs, shared by J1-J3; DQN-C 46 camera inputs (23,044 parameters), DQN-O 52 exact-state inputs (23,812 parameters) |
| DQN training | Double Q-learning, Huber loss, replay of 100,000 transitions, batch 64, one update per two transitions, target synchronisation every 500 updates, discount 0.99 per s (0.98 with the pressure reward); $\varepsilon$ from 1.0 to 0.05 over the first half of training; checkpoint every 25 episodes |
| DQN search | DQN-C, nine configurations: basic (46) or extended (52) camera features; reward from the junction queue, the queue with entry backlog and neighbour weight 0 or 0.5, or the junction pressure; $\Delta\in\{5,10\}$ s; minimum green 8 or 20 s; Adam $5\times10^{-4}$ (300 episodes) or $2.5\times10^{-4}$ (400 episodes); one configuration uses every DQN-O setting except the observation |
| Software | Python 3.11, NumPy; reverse-mode differentiation written for this study and gradient-checked (relative error below $10^{-6}$) |

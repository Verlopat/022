CAPTION: Table 1. Positioning of representative studies against the requirements of camera-based coordinated control for heterogeneous traffic.

| **Study group** | **Sensing input** | **Class-aware** | **Network coordination** | **Learning signal** | **Auditable timing rule** | **Evaluation** |
|---|---|---|---|---|---|---|
| Webster fixed time; SCATS | Historical counts; stop-line loops | No | SCATS subsystems | None | Yes | Analytical; field |
| Image-processing density control | Edge density per arm | No | No | None | Yes | Single junction |
| Single-junction and grid DRL (deep Q-learning, G-DQN, teacher-student, federated PPO, partial detection) | Simulator lane state; connected vehicles | No | No | Reward | No | Simulation (SUMO, CityFlow) |
| Multi-agent and hierarchical DRL (MARLIN-ATSC, MA2C, single-agent multi-junction, SHLight) | Simulator lane state | No | Joint or hierarchical policy | Reward | No | Simulation, up to 59 junctions |
| Graph and attention RL (GCN-RL, IG-RL, CoLight, AttendLight) | Simulator lane state | No | Graph message passing | Reward | No | Simulation, up to 3,971 junctions |
| Graph forecasting (T-GCN) | Loop and probe speeds | No | Graph convolution | Supervised | Not a controller | Real data |
| Vision sensing (YOLO counting, density CNN, congestion classifiers, edge monitoring) | Camera frames | Partly | No | Supervised | Not a controller | Real images |
| UVH-26 dataset and detectors | Bengaluru CCTV frames | Yes, 14 classes | No | Supervised | Not a controller | Real images |
| This study (GATC) | One camera frame per arm per cycle; optional 1 Hz stop-line check | Yes, PCU weights | GAT over arm and junction nodes | Supervised targets from vehicle logs | Yes, Eqs. (9)-(12) | Verified simulator, 5 scenarios × 20 seeds, camera and exact-state DQN baselines |

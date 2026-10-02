"""Static manuscript tables (markdown pipe tables with bold headers)."""
import json
import os

import numpy as np

import simcore as sc

os.makedirs('paper/tables', exist_ok=True)


def md_table(header, rows, caption):
    h = '| ' + ' | '.join(f'**{x}**' for x in header) + ' |'
    sep = '|' + '|'.join(['---'] * len(header)) + '|'
    body = ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return f'CAPTION: {caption}\n\n' + '\n'.join([h, sep] + body) + '\n'


def t_lit():
    header = ['Study group', 'Sensing input', 'Class-aware', 'Network coordination', 'Learning signal',
              'Auditable timing rule', 'Evaluation']
    rows = [
        ['Webster fixed time; SCATS', 'Historical counts; stop-line loops', 'No', 'SCATS subsystems', 'None',
         'Yes', 'Analytical; field'],
        ['Image-processing density control', 'Edge density per arm', 'No', 'No', 'None', 'Yes', 'Single junction'],
        ['Single-junction and grid DRL (deep Q-learning, G-DQN, teacher-student, federated PPO, partial detection)',
         'Simulator lane state; connected vehicles', 'No', 'No', 'Reward', 'No', 'Simulation (SUMO, CityFlow)'],
        ['Multi-agent and hierarchical DRL (MARLIN-ATSC, MA2C, single-agent multi-junction, SHLight)',
         'Simulator lane state', 'No', 'Joint or hierarchical policy', 'Reward', 'No', 'Simulation, up to 59 junctions'],
        ['Graph and attention RL (GCN-RL, IG-RL, CoLight, AttendLight)', 'Simulator lane state', 'No',
         'Graph message passing', 'Reward', 'No', 'Simulation, up to 3,971 junctions'],
        ['Graph forecasting (T-GCN)', 'Loop and probe speeds', 'No', 'Graph convolution', 'Supervised', 'Not a controller',
         'Real data'],
        ['Vision sensing (YOLO counting, density CNN, congestion classifiers, edge monitoring)', 'Camera frames',
         'Partly', 'No', 'Supervised', 'Not a controller', 'Real images'],
        ['UVH-26 dataset and detectors', 'Bengaluru CCTV frames', 'Yes, 14 classes', 'No', 'Supervised',
         'Not a controller', 'Real images'],
        ['This study (GATC)', 'One camera frame per arm per cycle; optional 1 Hz stop-line check', 'Yes, PCU weights',
         'GAT over arm and junction nodes', 'Supervised targets from vehicle logs', 'Yes, Eqs. (9)-(12)',
         'Verified simulator, 5 scenarios × 20 seeds, camera and exact-state DQN baselines'],
    ]
    return md_table(header, rows, 'Table 1. Positioning of representative studies against the requirements of camera-based coordinated control for heterogeneous traffic.')


def t_notation():
    header = ['Symbol', 'Meaning']
    rows = [
        ['$j$, $i$', 'Junction index (J1-J3) and arm index (W, N, E, S)'],
        ['$I_g$, $g_{\\min}$', 'Intergreen (5 s) and minimum green (8 s)'],
        ['$C_{j,k}$, $g_{ij,k}$', 'Cycle length and green time of arm $i$ in cycle $k$'],
        ['$L$, $S$', 'Lost time per cycle (24 s) and design saturation flow of a two-lane arm (1 PCU/s)'],
        ['$D$, $r^P_c$, $o^P$, $\\lambda^P$', 'Camera range (60 m), recall, occlusion factor and false-positive rate of profile $P$'],
        ['$w_c$, $s_{ij}$', 'PCU weight of group $c$ and PCU lane score'],
        ['$G=(V,E)$, $\\mathcal{N}(p)$', 'Junction graph with 15 nodes and neighbourhood of node $p$'],
        ['$\\alpha_{pq}$', 'Attention coefficient from node $p$ to node $q$'],
        ['$\\hat{Q}_{ij}$, $\\hat{A}_{ij}$', 'Predicted standing queue and arrivals in the next $H_t=120$ s (PCU)'],
        ['$\\hat{\\psi}_j$', 'Predicted junction flow ratio (impact score)'],
        ['$\\mu$, $\\beta$', 'Maximum-cycle scaling and cycle smoothing factor'],
        ['$\\kappa$, $\\rho_0$, $\\delta_{ij}$', 'Guard strength, guard threshold and guard factor'],
        ['$Q^{st}_m$, $\\kappa_s$', 'Storage of arm $m$ (car-spaces) and PCU per car-space (1.33)'],
        ['$Z_{ij}(t)$', 'Vehicles detected in the 20 m stop-line zone at time $t$'],
        ['$\\gamma$, $\\tau_k$, $r_k$, $Q_n$', 'DQN-C discount per second, transition duration, reward and queue normaliser (40 PCU)'],
        ['$\\bar{W}$, $J$', 'Average waiting time per vehicle and Jain fairness index'],
    ]
    return md_table(header, rows, 'Table 2. Notation.')


def t_classes():
    names = ['Two-wheeler', 'Bicycle', 'Three-wheeler', 'Car', 'LCV', 'Bus', 'Truck']
    uvh = ['Two-wheeler', 'Bicycle', 'Three-wheeler', 'Hatchback, Sedan, SUV, MUV', 'LCV, Van, Tempo-traveller, Other',
           'Bus, Mini-bus', 'Truck']
    header = ['Group', 'UVH-26 classes', 'Share (%)', 'PCU weight $w_c$', 'Headway (s/lane)', 'Jam space (car-spaces)',
              'Free speed (m/s)']
    rows = []
    for k in range(sc.NC):
        rows.append([names[k], uvh[k], f'{sc.MIX_BASE[k] * 100:.1f}', f'{sc.W_DESIGN[k]:g}', f'{sc.HEADWAY[k]:.1f}',
                     f'{sc.SPACE[k]:.1f}', f'{sc.SPEED_MU[k]:.1f} ± {sc.SPEED_SD[k]:.1f}'])
    return md_table(header, rows, 'Table 3. Vehicle groups, their UVH-26 source classes and simulation parameters (one car-space is 7 m of lane).')


def t_profiles():
    P = sc.make_profiles()
    header = ['Profile', 'Label set', 'Recall 2W / BIC / 3W / Car / LCV / Bus / Truck', 'Main confusion',
              'FP per frame', 'Occlusion $o^P$']
    conf = {'P0': 'None', 'P1': '5% to neighbouring groups',
            'P2': '3W: car 55%, motorcycle 20%, truck 25%; LCV: truck 60%, car 35%',
            'P3': '3W: car 50%, motorcycle 20%, truck 30%; LCV: truck 55%, car 40%'}
    rows = []
    for k in ['P0', 'P1', 'P2', 'P3']:
        p = P[k]
        rows.append([f'{k} {p.name}', '7 groups (14 UVH-26 classes)' if p.labels == 'uvh' else 'COCO: car, motorcycle, bicycle, bus, truck',
                     ' / '.join(f'{x:.2f}' for x in p.recall), conf[k], f'{p.fp_rate:.1f}', f'{p.occlusion:.2f}'])
    return md_table(header, rows, 'Table 4. Detector error profiles used by the perception model (modelling assumptions; see Section 4.6 for the sensitivity sweep).')


def t_hyper():
    hist = {t: json.load(open(f'models/{t}_hist.json')) for t in ['gat_full', 'gat_local', 'gcn_full', 'mlp_none']}
    header = ['Item', 'Setting']
    rows = [
        ['Layers and width', '2 message-passing layers, 4 heads x 16 units (GAT); 64 units (GCN, MLP); ELU'],
        ['Output heads', 'Arm: softplus for $(\\hat{Q},\\hat{A})$ scaled by $Y_s=40$ PCU; junction: sigmoid for $\\hat{\\psi}$'],
        ['Attention', 'LeakyReLU slope 0.2, masked softmax over $\\mathcal{N}(p)$ with self-loops'],
        ['Parameters', f"GAT {hist['gat_full']['metrics']['n_params']:,}; GCN {hist['gcn_full']['metrics']['n_params']:,}; MLP {hist['mlp_none']['metrics']['n_params']:,}"],
        ['Loss', 'Huber ($\\delta=0.5$) on normalised $Q$ and $A$; squared error on $\\psi$ with weight 2'],
        ['Gradient method', 'Adam ($\\beta_1=0.9$, $\\beta_2=0.999$), weight decay $10^{-4}$, batch 128'],
        ['Learning rate', 'Cosine schedule from $3\\times10^{-3}$ to $3\\times10^{-4}$'],
        ['Arm-node features (14)', 'PCU lane score; counts of four class groups; fill ratio of the camera zone; green share and length of the last cycle; score change since the previous frame; internal-arm and corridor-arm flags; green-now flag; time since the last green; node-type flag'],
        ['Junction-node features', 'Mean, maximum and spread of arm scores; total detections; mean fill ratio; cycle length; mean score change; middle-junction flag; elapsed cycle fraction; node-type flag (other entries zero)'],
        ['Behaviour policy', 'Cycle 60-160 s; greens from camera scores mixed with Dirichlet noise (weight 0-0.6); random or descending phase order; tactical layer in 60% of episodes'],
        ['Training distribution (estimators and DQN-C)', 'Demand factor 0.5-1.35; eastbound and westbound entry ratios 0.6-1.6 and 0.6-1.4; lane blockage in 30% of episodes (random arm and lane, 600-1,500 s); heavy-vehicle mix in 20%; profiles P0/P1/P2/P3 with probabilities 0.10/0.60/0.15/0.15'],
        ['Epochs', '100 maximum, early stopping after 20 epochs without a lower held-out loss'],
        ['Data', '400 episodes, 26,745 graph snapshots, episode-level split 80/20'],
        ['Training time (CPU, one core)', f"GAT {hist['gat_full']['metrics']['train_time_s'] / 60:.1f} min; GCN {hist['gcn_full']['metrics']['train_time_s'] / 60:.1f} min; MLP {hist['mlp_none']['metrics']['train_time_s'] / 60:.1f} min; DQN-C {json.load(open('models/dqn_d10_s0_hist.json'))['train_time_s'] / 60:.1f} min; DQN-O {json.load(open('models/dqno_d10_g8_hist.json'))['train_time_s'] / 60:.1f} min"],
        ['DQN-C and DQN-O network', DQN_NET],
        ['DQN training', DQN_TRAIN],
        ['DQN search', DQN_SEARCH],
        ['Software', 'Python 3.11, NumPy; reverse-mode differentiation written for this study and gradient-checked (relative error below $10^{-6}$)'],
    ]
    return md_table(header, rows, 'Table 5. Configuration and training protocol of the estimators and of the DQN baselines.')


DQN_NET = ('Two hidden layers of 128 ReLU units, 4 outputs, shared by J1-J3; DQN-C 46 camera inputs (23,044 parameters), '
           'DQN-O 52 exact-state inputs (23,812 parameters)')
DQN_TRAIN = ('Double Q-learning, Huber loss, replay of 100,000 transitions, batch 64, one update per two transitions, '
             'target synchronisation every 500 updates, discount 0.99 per s (0.98 with the pressure reward); $\\varepsilon$ from 1.0 to 0.05 over the first '
             'half of training; checkpoint every 25 episodes')
DQN_SEARCH = ('DQN-C, nine configurations: basic (46) or extended (52) camera features; reward from the junction queue, '
              'the queue with entry backlog and neighbour weight 0 or 0.5, or the junction pressure; $\\Delta\\in\\{5,10\\}$ s; '
              'minimum green 8 or 20 s; Adam $5\\times10^{-4}$ (300 episodes) or $2.5\\times10^{-4}$ (400 episodes); one configuration uses every DQN-O setting except the observation')


def t_tuning():
    cfg = json.load(open('results/final_configs.json'))
    header = ['Controller', 'Sensing', 'Search space (seeds 500-505)', 'Frozen setting']
    rows = [
        ['FT-W', 'None', '$C_{\\max}\\in\\{160,200,240\\}$ s', 'Design-day Webster plan; peak cycle 140 s, off-peak 70 s'],
        ['FT-W*', 'None (true scenario flows)', '$C_{\\max}\\in\\{160,200,240\\}$ s', '$C_{\\max}=200$ s'],
        ['MP-C', 'Own and receiving arms, once per junction cycle', '$C\\in\\{80,100,120,150\\}$ s', '$C=120$ s'],
        ['LPS', 'Own 4 arms, once per junction cycle', '$C\\in\\{90,120,150\\}$ s', '$C=120$ s'],
        ['GATC-S', 'All 12 arms, once per junction cycle', '$\\kappa$, $\\rho_0$, $\\mu$, $C_{\\max}$: 72 settings', '$\\kappa=0.5$, $\\rho_0=0.5$, $\\mu=1.0$, $C_{\\max}=240$ s'],
        ['ACT', 'Stop-line loops, every lane, continuous', 'max green 60-120 s, gap 2-3 s', 'max green 90 s, gap 2.0 s'],
        ['MP-A', 'Own and receiving arms every $\\Delta$ s', '$\\Delta\\in\\{10,15,20,30\\}$ s, min green 8-30 s', '$\\Delta=20$ s, min green 20 s'],
        ['LPS-T', 'As LPS, plus 1 Hz green-arm check', '$C\\in\\{200,240,300\\}$ s', '$C=240$ s'],
        ['DQN-C', 'Own and downstream arms every $\\Delta$ s', 'Nine configurations (Table 5) × checkpoints', 'Basic features, junction-queue reward, $\\Delta=10$ s, minimum green 8 s, episode 225 of 300'],
        ['GATC', 'As GATC-S, plus 1 Hz green-arm check', '$\\kappa$, $\\mu$, $C_{\\max}$: 24 settings', '$\\kappa=0.5$, $\\rho_0=0.7$, $\\mu=2.0$, $C_{\\max}=240$ s'],
        ['DQN-O (reference)', 'Exact simulator state every $\\Delta$ s', 'Checkpoints of one configuration', 'Queue reward with entry backlog, $\\Delta=10$ s, minimum green 8 s, episode 200 of 400'],
    ]
    return md_table(header, rows, 'Table 6. Controllers, sensing regimes and settings frozen after tuning on seeds disjoint from the test seeds.')


if __name__ == '__main__':
    out = dict(lit=t_lit(), notation=t_notation(), classes=t_classes(), profiles=t_profiles(), hyper=t_hyper(),
               tuning=t_tuning())
    for k, v in out.items():
        open(f'paper/tables/{k}.md', 'w').write(v)
    print('written', list(out))

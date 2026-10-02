# Methods

## Study Design and Problem Statement

The study is a controlled simulation experiment with paired random seeds. A corridor of $N=3$ signalised junctions is indexed by $j$, and each junction has four arms $i\in\{W,N,E,S\}$ with two lanes each; an arm is the inbound road section of one direction, so the corridor has $N\times L=12$ arms. Figure 1 shows the online loop and the offline training. Every junction runs split phasing: one phase serves one arm with all its movements and is followed by an intergreen of $I_g=5$ s (3 s amber, 2 s all-red). Cycle $k$ of junction $j$ lasts $C_{j,k}=\sum_{i}(g_{ij,k}+I_g)$ with green times $g_{ij,k}\ge g_{\min}=8$ s. Using cameras only, the controller chooses $C_{j,k}$, the greens and the phase order to minimise the average waiting time per vehicle

$$\bar{W}=\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}\left(e_v+\sum_{(i,j)\in\mathcal{R}_v}\left(t^{\mathrm{dis}}_{v,ij}-t^{\mathrm{ff}}_{v,ij}\right)\right),\qquad(1)$$

where $\mathcal{V}$ is the measured cohort, $e_v$ the entry delay at the boundary, $\mathcal{R}_v$ the arms on the route of $v$, $t^{\mathrm{ff}}$ the free-flow arrival at the stop line and $t^{\mathrm{dis}}$ the discharge time. Two sensing regimes are compared. In the snapshot regime a controller receives one frame per arm when a junction starts a cycle, the budget of the original CPU prototype. In the continuous regime it may also read camera frames within the cycle. Table 2 lists the notation.

FIGURE:01

TABLE:notation

## Corridor, Traffic Population and Scenarios

The corridor reproduces the three-junction, dual-lane layout of the original prototype (Figure 2): junctions 300 m and 250 m apart, 300 m boundary entry links, left-hand traffic and turning shares of 0.15/0.72/0.13 (left/through/right) on corridor arms and 0.30/0.40/0.30 on side streets. Vehicles belong to seven groups formed from the 14 UVH-26 classes, in a two-wheeler-dominated mix (Figure 3). Groups differ in free-flow speed, jam spacing and saturation headway (Table 3), and headways vary per vehicle with a log-normal factor (coefficient of variation 0.12). Arrivals at the eight boundary arms are non-homogeneous Poisson processes whose base flows give a critical flow ratio of $Y=0.70$ at the middle junction, modulated by a time-of-day multiplier that rises from 0.85 to 1.15 and back (Figure 4). Five scenarios span the operating range: S1 off-peak (0.60 × base), S2 peak (1.00 ×), S3 oversaturated peak (1.30 ×), S4 tidal surge (eastbound entry 1.5 ×, westbound 0.6 ×) and S5 incident (S2 demand with the near-side lane of the J3 west arm blocked from 1,500 s to 2,700 s). Runs last 4,800 s; vehicles generated between 600 s and 3,600 s form the cohort, and those still inside at 4,800 s contribute their delay so far.

FIGURE:02

FIGURE:03

FIGURE:04

## Perception and PCU Lane Scoring

Each camera covers $D=60$ m upstream of the stop line. A vehicle of group $c$ at depth $z$ is detected with probability

$$P(\mathrm{detect}\mid c,z)=r^{P}_{c}\left(1-o^{P}\min\left(1,z/D\right)\right),\qquad(2)$$

receives a label drawn from the confusion row $\Pi^{P}(c,\cdot)$, and false positives arrive at rate $\lambda^{P}$ per frame. Profile P0 is an oracle, P1 the UVH-26-trained YOLO checkpoint (best.pt), and P2 and P3 the COCO-trained YOLOv8m and YOLOv8n, which lack three-wheeler, LCV, van and tempo-traveller classes (Table 4). The profile values are modelling assumptions scaled to the gain reported for UVH-26-trained over COCO-trained detectors (8.4% to 31.5% in mAP50:95); Section 4.6 sweeps recall and false positives well beyond them. Detections become a PCU lane score

$$s_{ij}=\sum_{c\in\mathcal{K}}w_{c}\,\hat{n}_{ij,c},\qquad(3)$$

with design weights of 0.5 (two-wheeler, bicycle), 1.0 (three-wheeler, car), 1.5 (LCV) and 3.0 (bus, truck). Discharge in the simulator follows class headways, so the weights only approximate true discharge time, as in field practice.

TABLE:classes

TABLE:profiles

## Junction Graph and GAT Estimator

The corridor state is a graph $G=(V,E)$ with $|V|=N\times L+N=15$ nodes, 12 arm nodes and 3 junction nodes (Figure 5), joined by intra-junction edges (36 directed), arm-hub edges (24), hub-hub trunk edges (6), feeder edges between each internal arm and the upstream arms that feed it (24) and self-loops. Every node carries a 14-dimensional feature vector built from the latest frames and the signal history (Table 5).

FIGURE:05

Two masked multi-head attention layers follow the GAT formulation. For head $h$ of layer $\ell$ and neighbour $q\in\mathcal{N}(p)$,

$$\alpha^{(\ell,h)}_{pq}=\frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}_{\mathrm{s}}\mathbf{W}\mathbf{h}_{p}+\mathbf{a}^{\top}_{\mathrm{d}}\mathbf{W}\mathbf{h}_{q}\right)\right)}{\sum_{r\in\mathcal{N}(p)}\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}_{\mathrm{s}}\mathbf{W}\mathbf{h}_{p}+\mathbf{a}^{\top}_{\mathrm{d}}\mathbf{W}\mathbf{h}_{r}\right)\right)},\qquad(4)$$

$$\mathbf{h}^{(\ell+1)}_{p}=\big\Vert_{h=1}^{4}\,\mathrm{ELU}\left(\sum_{q\in\mathcal{N}(p)}\alpha^{(\ell,h)}_{pq}\mathbf{W}^{(\ell,h)}\mathbf{h}^{(\ell)}_{q}+\mathbf{b}^{(\ell,h)}\right),\qquad(5)$$

with head-specific $\mathbf{W}$, $\mathbf{a}_{\mathrm{s}}$ and $\mathbf{a}_{\mathrm{d}}$ in Eq. (4). Output heads read the second-layer embedding joined to the input features:

$$\left(\hat{Q}_{ij},\hat{A}_{ij}\right)=Y_s\,\mathrm{softplus}\left(\mathbf{W}_{y}\left[\mathbf{h}^{(2)}_{p}\Vert\mathbf{x}_{p}\right]+\mathbf{b}_{y}\right),\qquad \hat{\psi}_{j}=\sigma\left(\mathbf{w}^{\top}_{\psi}\left[\mathbf{h}^{(2)}_{j}\Vert\mathbf{x}_{j}\right]+b_{\psi}\right).\qquad(6)$$

$\hat{Q}_{ij}$ is the standing queue of the arm in PCU, including the part beyond the camera view, $\hat{A}_{ij}$ the PCU arriving within the next $H_t=120$ s, $\hat{\psi}_j$ the junction flow ratio, which serves as the junction impact score, and $Y_s=40$ PCU. Targets come from vehicle passage logs:

$$Q_{ij}(t)=\sum_{v:\,t^{\mathrm{ff}}_{v}\le t<t^{\mathrm{dis}}_{v}}w_{c(v)},\qquad A_{ij}(t)=\sum_{v:\,t<t^{\mathrm{ff}}_{v}\le t+H_t}w_{c(v)},\qquad \psi_{j}(t)=\min\left(1,\frac{\sum_{i}A_{ij}(t)}{S\,H_t}\right),\qquad(7)$$

with $S=1$ PCU/s per two-lane arm. The loss is

$$\mathcal{L}=\frac{1}{24}\sum_{i,j}\sum_{u\in\{Q,A\}}\mathrm{Huber}_{0.5}\left(\frac{\hat{u}_{ij}-u_{ij}}{Y_s}\right)+\frac{2}{3}\sum_{j}\left(\hat{\psi}_{j}-\psi_{j}\right)^{2}.\qquad(8)$$

Training data come from 400 episodes of an exploratory behaviour policy with random cycles, splits and phase orders, run over randomised demand levels, tidal ratios, lane blockages, vehicle mixes and detector profiles (Table 5). The 26,745 graph snapshots are split by episode into 80% for training and 20% held out. GCN, GAT-local (without trunk and feeder edges) and graph-free MLP estimators of equal width are trained identically (Table 5), each from three random initialisations.

TABLE:hyper

## Timing Law

At the end of each cycle, the strategic layer evaluates the estimator on fresh frames and applies three explicit rules. The cycle follows Webster's formula at the predicted flow ratio,

$$C_{j,k}=\mathrm{round}_{5}\left(\beta C_{j,k-1}+(1-\beta)\min\left(C_{\max},\max\left(C_{\min},\ \mu\,\frac{1.5L+5}{1-\hat{\psi}_{j}}\right)\right)\right),\qquad(9)$$

with lost time $L=24$ s, $\beta=0.5$, $C_{\min}=60$ s and a scaling $\mu$ that converts the fixed-time cycle into a maximum cycle when the tactical layer is active. A spillback guard discounts arms whose movements feed a loaded downstream arm $m$ with storage $Q^{\mathrm{st}}_{m}$ car-spaces:

$$\delta_{ij}=1-\kappa\sum_{u\in\{L,T,R\}}\pi_{ij,u}\,\min\left(1,\max\left(0,\frac{\rho_{m(u)}-\rho_{0}}{1-\rho_{0}}\right)\right),\qquad \rho_{m}=\frac{\hat{Q}_{m}+\hat{A}_{m}}{\kappa_{s}Q^{\mathrm{st}}_{m}},\qquad(10)$$

where $\pi_{ij,u}$ is the turning share of movement $u$ and $\kappa_s=1.33$ PCU per car-space. Green follows the guarded demand over the planned cycle,

$$g_{ij}=g_{\min}+\left(C_{j,k}-4I_g-4g_{\min}\right)\frac{\delta_{ij}\left(\hat{Q}_{ij}+\hat{A}_{ij}C_{j,k}/H_t\right)}{\sum_{l}\delta_{lj}\left(\hat{Q}_{lj}+\hat{A}_{lj}C_{j,k}/H_t\right)},\qquad(11)$$

and phases run in descending order of guarded demand. In the continuous regime, the tactical layer inspects the 20 m stop-line zone of the green arm once per second after $g_{\min}$ and ends the phase when two consecutive checks find no vehicle, or when vehicles are present but none has crossed for 4 s:

$$\left(Z_{ij}(t)=0\wedge Z_{ij}(t-1)=0\right)\ \vee\ \left(Z_{ij}(t)>0\wedge t-t^{\mathrm{last}}_{ij}>4\ \mathrm{s}\right).\qquad(12)$$

The first condition removes idle green; the second removes green blocked by spillback or an incident. Unused green passes to the remaining phases of the cycle in proportion to their planned greens. Algorithm 1 gives one cycle. The two operating modes are named GATC-S (strategic layer only) and GATC (both layers).

ALGORITHM:1

## Simulator, Baselines and Protocol

A mesoscopic spatial-queue simulator advances in 1 s steps with event-accurate discharge. Left movers use the near-side lane, right movers the off-side lane and through movers the shorter lane. A queued vehicle crosses at

$$\tau_{v}=\max\left(\tau_{v^{-}}+h_{v},\,t^{\mathrm{ff}}_{v},\,t\right),\qquad \tau_{v}-h_{v}/2\le t^{\mathrm{g,end}},\qquad(13)$$

where $\tau_{v^{-}}$ is the previous crossing in the lane or green start plus 2 s start-up lost time, $h_v$ the vehicle headway and $t^{\mathrm{g,end}}$ the end of effective green, 1 s into amber. A crossing is admitted only if the receiving link has free storage, which produces blocked green and spillback. The simulator was first checked against Webster's delay formula at an isolated junction (cars only, 96 s cycle, 19 s greens, 16 runs of 8 h per degree of saturation).

Eleven controllers are compared (Table 6), two of them references with information no camera provides. Without sensing: FT-W, a Webster plan from design-day demand (off-peak plan in S1, peak plan in S2-S5), and the reference FT-W*, built from the true mean flows. In the snapshot regime: MP-C, cyclic max-pressure with pressure-proportional splits; LPS, the original prototype's proportional split of Eq. (3) with a user-set cycle; and GATC-S. In the continuous regime: ACT, fully actuated control with presence-mode stop-line loops; MP-A, acyclic max-pressure re-evaluated from frames every $\Delta$ s; LPS-T, LPS with the tactical layer of Eq. (12); DQN-C; and GATC.

DQN-C is a deep reinforcement learning baseline on the same camera frames, with one Q-network (two hidden layers of 128 ReLU units) shared by the three junctions. After the minimum green and then every $\Delta$ s, a junction observes camera features of its four arms and downstream internal arms, the current phase, elapsed green and a junction indicator, and selects the arm to serve next (maximum green 90 s). Double Q-learning on transitions of duration $\tau_k$ uses the target

$$y_k=r_k+\gamma^{\tau_k}\,Q^{-}\!\left(\mathbf{o}_{k+1},\arg\max_{a}Q(\mathbf{o}_{k+1},a)\right),\qquad r_k=-\frac{1}{5Q_n}\int_{t_k}^{t_k+\tau_k}\sum_{i}q_{ij}(t)\,dt,\qquad(14)$$

where $q_{ij}$ is the queued PCU of arm $i$ read from the simulator during training only, $\gamma=0.99$ per second and $Q_n=40$ PCU. The reference DQN-O has the same design but observes the exact simulator state (full queues, link occupancy, entry backlog), and its reward also counts the entry backlog.

Every controller was tuned on seeds 500-505 over all scenarios, and the setting with the lowest mean $\bar{W}$ was frozen; for DQN-C this covered nine training configurations and their checkpoints (Table 5). Gap-type thresholds were bounded below by the 2 s of field practice, because the simulator has no dilemma-zone model. Testing uses seeds 1-20 with common random numbers, so demand, vehicle attributes and routes are identical across controllers within a seed.

TABLE:tuning

## Metrics and Analysis Plan

The primary metric is $\bar{W}$. Secondary metrics are travel time, throughput, time-averaged queue per arm (PCU), the 95th percentile of vehicle waiting time, Jain's fairness index over the 12 arm delays $d_a$,

$$J=\frac{\left(\sum_{a}d_{a}\right)^{2}}{12\sum_{a}d_{a}^{2}},\qquad(15)$$

the green utilisation ratio GUR (share of effective green lane-seconds with vehicles discharging) and the blocked-green share. Paired differences over the 20 seeds are tested with the two-sided Wilcoxon signed-rank test, p-values are adjusted by the Holm procedure across all 50 comparisons at $\alpha=0.05$, the matched-pairs rank-biserial correlation $r_{rb}$ is the effect size, and 95% confidence intervals come from 10,000 bootstrap resamples. Estimator accuracy is reported on the held-out episodes as mean absolute error (MAE) and $R^2$.

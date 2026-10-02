# Results

## Simulator Verification

Figure 6 compares the simulator with Webster's delay formula at an isolated junction. From a degree of saturation of 0.30 to 0.90, the simulated mean delay stays within 2.1% of the formula: 34.3 s against 34.1 s at $x=0.30$ and 53.8 s against 55.0 s at $x=0.90$. At $x=0.95$ the simulator gives 76.1 s against 81.1 s (-6.2%). This shortfall has the expected sign, because the steady-state random-delay term of the formula exceeds the delay that accumulates over a finite 8 h horizon. The discharge, lost-time and queue logic reproduce the reference model over the range used in the controller study.

FIGURE:06

## Estimator Accuracy

Table 7 and Figures 7 and 8 report accuracy on the 80 held-out episodes, with learned estimators averaged over three training seeds. The raw snapshot score explains none of the variance of next-cycle demand ($R^2=-0.20$) because it sees 60 m of the arm and no arrivals. A least-squares calibration raises $R^2$ to 0.61, but its predictions flatten above about 80 PCU, where queues leave the camera view (Figure 8). The learned estimators do much better, and the GAT is the most accurate: MAE of 7.58 PCU and $R^2$ of 0.839, against 8.62 and 0.798 for the MLP. The gain concentrates where neighbour information matters. On internal arms, $R^2$ rises from 0.636 (MLP) and 0.670 (GAT-local) to 0.726, and for the junction flow ratio from 0.520 to 0.656. Queue errors of the MLP and the GAT are close (5.24 and 5.14 PCU), whereas the arrival error falls from 5.78 to 5.13 PCU. H1 holds: graph attention improves mainly the part of demand that originates at neighbouring junctions.

Accuracy and control outcome do not rank the estimators in the same order. When each estimator drives GATC-S, the MLP gives the lowest average waiting time (130.2 ± 1.4 s/veh across training seeds), followed by the GCN (134.2 ± 1.0), the GAT (135.2 ± 2.6) and GAT-local (138.3 ± 2.0). With the tactical layer, all four stay between 90.9 and 91.5 s/veh. Every learned estimator improves on raw camera scores by a wide margin in the snapshot regime, but the extra accuracy of graph attention does not carry over into lower waiting time in this corridor.

TABLE:estimator

FIGURE:07

FIGURE:08

## Waiting Time Across Controllers and Scenarios

Table 8 and Figure 9 give the average waiting time, Table 9 the paired tests and Figure 10 the relative differences. In the snapshot regime, GATC-S reduced waiting time against LPS by 30.5%, 8.5%, 14.2% and 34.5% in S1-S4 (Holm-adjusted $p<0.001$, $|r_{rb}|\ge0.99$). Against design-day FT-W it was faster in S1, S2 and S4 by 12.2%, 12.4% and 41.0%. Two results run the other way: GATC-S was 5.8% slower than FT-W in S3, and in S5 it was 12.8% slower than LPS and 14.9% slower than FT-W. H2 holds in four of five scenarios against LPS and in three of five against fixed time. Most of the gain comes from cycle adaptation: GATC-S ran mean cycles of 70 s in S1 and 181 s in S3, where LPS kept 120 s. MP-C was the weakest controller (253.7 s/veh on average), because pressures computed from truncated snapshots under-serve the long corridor queues.

TABLE:main

FIGURE:09

In the continuous regime, GATC was faster than loop-actuated ACT by 5.0%, 15.0% and 10.1% in S2, S3 and S4 ($p<0.001$) and level with it in S1 and S5 (-0.9% and -1.0%, $p=1.0$). Against camera-based MP-A it was faster by 16.6%, 7.6%, 30.9% and 13.4% in S1-S4 and level in S5. Relative to the original LPS controller, GATC cut waiting time by 26.1% to 55.8%, and relative to design-day fixed time by 19.0% to 60.2%. Even FT-W* averaged 125.0 s/veh against 91.3 s/veh for GATC. The comparison with LPS-T qualifies this result: LPS-T was 2.5% and 2.2% faster than GATC in S1 and S2 ($p=0.033$ and $0.030$), GATC was 3.2% faster in S3 ($p=0.016$), and S4 and S5 showed no detectable difference. Once the stop-line check is present, the strategic estimator shifts waiting time by less than 4% in either direction.

TABLE:stats

FIGURE:10

The deep Q-network trained on camera frames behaved differently at low and high demand. DQN-C gave the lowest waiting time of all deployable controllers in S1 (45.0 s/veh; GATC 11.1% slower), but it kept greens short at every demand level (mean 10.4 s in S1, 12.9 s in S2 and 15.5 s in S3), lost capacity to intergreens and left some arms waiting: 158.5-481.2 s/veh in S2-S5, where GATC was 51.1-69.6% faster. None of the nine camera configurations avoided this collapse, including one trained exactly like DQN-O except for its observation (Figure 11). The same network with the exact simulator state, DQN-O, converged within 100 episodes and averaged 83.9 s/veh over S1-S5, 4.2-16.1% below GATC. Measured on the S1-S5 mean, GATC recovered 94% of the difference between camera-trained and exact-state reinforcement learning. H3 holds against ACT, MP-A and DQN-C, with ties in S1 and S5 against ACT and the off-peak exception against DQN-C.

FIGURE:11

## Queue Dynamics, Spillback and Fairness

Figure 12 traces the queued demand of all arms in S5. Under FT-W and LPS the queue rose from about 100 PCU to 250-280 PCU during the blockage and receded slowly after the lane reopened. GATC-S and DQN-C peaked near 370 PCU and recovered last. The strategic plan of GATC-S keeps assigning long greens to arms whose predicted demand is high but whose vehicles cannot move, so its blocked-green share reached 5.7% and a link stood at least 95% full during 30.9% of the time. GATC, ACT and MP-A held the peak near 190-210 PCU and cleared within about 10 min of reopening, because condition two of Eq. (12) releases a stalled green after 4 s. In S3, GATC delivered the highest throughput among deployable controllers (5,947 veh/h) and the lowest 95th-percentile waiting time (331 s; Table 10), and its waiting-time distribution lies left of all deployable controllers except LPS-T (Figure 13). Its Jain index in S3 was 0.975, level with ACT (0.972) and well above MP-A (0.764) and DQN-O (0.847), whose rules favour total throughput over low-volume side arms; in S5 it was 0.870 against 0.919 for ACT.

FIGURE:12

TABLE:secondary

FIGURE:13

## Ablation

Table 11 and Figure 14 remove one design element at a time; estimator variants were compared in Section 4.2. For GATC-S, fixing the cycle at 120 s raised waiting time by 24.4% on average and by 51.1% in S3, and replacing the estimator with raw camera scores (which yields LPS) raised it by 22.2%. Removing the spillback guard added 2.5%, mostly in S3 (+8.9%). Descending-demand phase order, inherited from the original prototype, helped in S1, S2 and S4 but hurt in S3 and S5, so fixed order was 4.2% faster on average. For GATC, every strategic change stayed within 2% on average: removing the guard changed waiting time by 0.0%, fixing the cycle by +0.1% and fixing the phase order by +1.5%, while camera scores in place of the estimator (LPS-T) changed it by -0.9%. The tactical layer absorbs most errors of the strategic plan. PCU weighting versus raw counts changed LPS waiting time by 1.6% in S2 ($p=0.02$) and by -0.6% in the heavy-vehicle mix S2HV ($p=0.78$).

TABLE:ablation

FIGURE:14

## Detector Quality

Table 12 compares the four detector profiles. Controllers in the snapshot regime performed worse with COCO-trained detectors: from P1 to P3, waiting time rose by 15.5% for LPS, 25.3% for MP-A and 11.8% for GATC-S. GATC-S was slower with the oracle profile P0 (140.4 s/veh) than with P1 (132.9 s/veh), because the estimator learned to compensate the P1 miss rate, so error-free counts fall outside its calibration. DQN-C varied between 212.9 and 325.8 s/veh and was worst with P0, which removes detection errors but not the 60 m limit of the view. LPS-T and GATC varied by less than 1% across profiles, because the tactical check needs only vehicle presence near the stop line. Figure 15 sweeps the P1 recall and the false-positive rate for GATC-S in S2. At nominal recall each extra false positive per frame added 10-12 s/veh (95.8 to 128.6 s/veh at three extra detections), while halving recall without false positives more than doubled waiting time (209.5 s/veh). Combinations whose errors offset, such as recall ×0.7 with one or two extra detections, stayed at the nominal level. The estimator responds to the net bias of the PCU count rather than to recall alone.

TABLE:perception

FIGURE:15

## Attention Pattern and Computational Cost

Figure 16 shows the mean second-layer attention of GATC-S in S4. Internal arms attend to the arms that feed them: J1-E and J3-W place 0.18-0.20 of their attention on the side arms of J2, whose left and right turners enter both corridor directions, and J2-W spreads its attention over J1-W, J1-N and J1-S. The J2 junction node concentrates on its own side arms (about 0.20 each). The pattern follows the physical routing of the corridor and gives operators a check on what the estimator uses. The estimator has 5,613 parameters and needs about 0.23 MFLOP and 0.29 ms per evaluation on one CPU core (Table 13). Camera processing dominates the cost: GATC-S reads about 1,220 frames per hour for the corridor, GATC adds about 5,100 stop-line checks, and DQN-C reads 4,807 frames per hour.

FIGURE:16

TABLE:cost

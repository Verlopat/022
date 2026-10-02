CAPTION: Table 8. Average waiting time per vehicle (s/veh, mean ± SD over 20 paired seeds). † marks the lowest value among deployable controllers; FT-W* (true scenario flows) and DQN-O (exact simulator state) are references that no camera deployment can reach.

| **Controller** | **Sensing** | **S1** | **S2** | **S3** | **S4** | **S5** | **Mean** |
|---|---|---|---|---|---|---|---|
| FT-W | None | 61.7 ± 1.0 | 107.2 ± 2.8 | 198.2 ± 15.3 | 199.6 ± 26.4 | 164.6 ± 10.8 | 146.3 |
| FT-W* | None (true flows) | 61.7 ± 1.0 | 107.2 ± 2.8 | 171.3 ± 9.8 | 120.4 ± 3.3 | 164.6 ± 10.8 | 125.0 |
| MP-C | Snapshot | 106.3 ± 6.3 | 166.7 ± 16.4 | 446.2 ± 37.7 | 341.7 ± 46.4 | 207.5 ± 17.5 | 253.7 |
| LPS | Snapshot | 78.0 ± 1.6 | 102.6 ± 3.8 | 244.5 ± 21.8 | 179.7 ± 16.4 | 167.6 ± 31.9 | 154.5 |
| GATC-S | Snapshot | 54.2 ± 1.6 | 93.9 ± 4.7 | 209.8 ± 16.0 | 117.7 ± 12.5 | 189.1 ± 16.4 | 132.9 |
| ACT | Continuous (loops) | 50.5 ± 2.2 | 79.8 ± 3.1 | 172.3 ± 15.9 | 88.4 ± 4.5 | 105.8 ± 9.8 | 99.3 |
| MP-A | Continuous | 59.9 ± 1.5 | 82.0 ± 2.6 | 211.9 ± 22.1 | 91.7 ± 6.4 | 107.4 ± 6.8 | 110.6 |
| DQN-C | Continuous | 45.0 ± 1.0 † | 158.5 ± 42.3 | 481.2 ± 52.6 | 165.4 ± 32.3 | 214.3 ± 46.9 | 212.9 |
| LPS-T | Continuous | 48.8 ± 1.5 | 74.2 ± 2.5 † | 151.1 ± 11.1 | 80.0 ± 3.7 | 100.9 ± 7.6 † | 91.0 |
| GATC | Continuous | 50.0 ± 1.5 | 75.8 ± 2.2 | 146.4 ± 10.0 † | 79.4 ± 4.2 † | 104.7 ± 9.1 | 91.3 |
| DQN-O | Exact state | 46.1 ± 1.3 | 72.4 ± 2.9 | 134.5 ± 10.5 | 76.2 ± 2.6 | 90.2 ± 4.1 | 83.9 |

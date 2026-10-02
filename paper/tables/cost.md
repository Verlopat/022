CAPTION: Table 13. Sensing and computing budget per hour for the three-junction corridor (means over S1-S5, seed 1). Detector costs use published figures for 640-pixel frames: YOLO11-S 21.6 GFLOP and 90.0 ms (CPU, ONNX), YOLOv8n 8.7 GFLOP. The GAT estimator adds 0.23 MFLOP and 0.29 ms per call (5,613 parameters); network calls for DQN-C are Q-network evaluations.

| **Controller** | **Frames per hour** | **Stop-line checks per hour** | **GFLOP/s, all YOLO11-S** | **GFLOP/s, checks on YOLOv8n** | **CPU s per hour, YOLO11-S** | **Network calls per hour** |
|---|---|---|---|---|---|---|
| LPS | 360 | 0 | 2.2 | 2.2 | 32 | - |
| MP-C | 480 | 0 | 2.9 | 2.9 | 43 | - |
| GATC-S | 1,220 | 0 | 7.3 | 7.3 | 110 | 102 |
| MP-A | 2,368 | 0 | 14.2 | 14.2 | 213 | - |
| LPS-T | 479 | 5,111 | 33.5 | 15.2 | 503 | - |
| DQN-C | 4,807 | 0 | 28.8 | 28.8 | 433 | 902 |
| GATC | 1,436 | 5,112 | 39.3 | 21.0 | 589 | 120 |

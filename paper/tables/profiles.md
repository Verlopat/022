CAPTION: Table 4. Detector error profiles used by the perception model (modelling assumptions; see Section 4.6 for the sensitivity sweep).

| **Profile** | **Label set** | **Recall 2W / BIC / 3W / Car / LCV / Bus / Truck** | **Main confusion** | **FP per frame** | **Occlusion $o^P$** |
|---|---|---|---|---|---|
| P0 Oracle | 7 groups (14 UVH-26 classes) | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 | None | 0.0 | 0.00 |
| P1 UVH-26 YOLO (best.pt) | 7 groups (14 UVH-26 classes) | 0.88 / 0.78 / 0.93 / 0.95 / 0.90 / 0.95 / 0.93 | 5% to neighbouring groups | 0.3 | 0.15 |
| P2 COCO YOLOv8m | COCO: car, motorcycle, bicycle, bus, truck | 0.82 / 0.70 / 0.70 / 0.93 / 0.80 / 0.90 / 0.88 | 3W: car 55%, motorcycle 20%, truck 25%; LCV: truck 60%, car 35% | 0.5 | 0.25 |
| P3 COCO YOLOv8n | COCO: car, motorcycle, bicycle, bus, truck | 0.66 / 0.52 / 0.55 / 0.87 / 0.68 / 0.84 / 0.80 | 3W: car 50%, motorcycle 20%, truck 30%; LCV: truck 55%, car 40% | 0.7 | 0.35 |

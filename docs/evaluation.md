# Model evaluation

- **Date:** 2026-11-10
- **Model:** `models/pcb_defect_v1.onnx` (ONNX Runtime, CPU), same preprocessing as the API
- **Dataset:** DeepPCB (Kaggle repackaging, 1,200/150/150 split)
- **Matching:** a detection is correct if it has the right class and IoU >= 0.5 with an unmatched labelled box (IoU 0.50-0.95 for mAP50-95); AP uses 101-point interpolation

## test split (`data/DeepPCB/test`)

150 images, 984 labelled defects.

**Ultralytics-style** (conf 0.001, NMS IoU 0.7; P and R at the best-F1 confidence):

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 150 | 984 | 0.976 | 0.943 | 0.978 | 0.799 |
| Open | 134 | 191 | 0.973 | 0.949 | 0.987 | 0.742 |
| Short | 114 | 170 | 0.989 | 0.876 | 0.953 | 0.714 |
| Mousebite | 129 | 194 | 0.979 | 0.959 | 0.978 | 0.775 |
| Spur | 113 | 153 | 0.984 | 0.948 | 0.977 | 0.757 |
| Copper | 119 | 140 | 0.986 | 0.964 | 0.992 | 0.917 |
| Pin-hole | 124 | 136 | 0.942 | 0.963 | 0.980 | 0.886 |

**API settings** (conf 0.25, NMS IoU 0.45, all boxes kept): P = 0.926, R = 0.961, F1 = 0.943

**API's single answer per image** (what `/inspect` returns today): the reported defect type is actually present on the board in 150/150 images (100.0%); `no_defect_detected` in 0.

## valid split (`data/DeepPCB/valid`)

150 images, 1005 labelled defects.

**Ultralytics-style** (conf 0.001, NMS IoU 0.7; P and R at the best-F1 confidence):

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 150 | 1005 | 0.970 | 0.959 | 0.989 | 0.795 |
| Open | 138 | 189 | 0.995 | 0.979 | 0.994 | 0.716 |
| Short | 117 | 167 | 0.994 | 0.921 | 0.989 | 0.727 |
| Mousebite | 126 | 189 | 0.958 | 0.947 | 0.979 | 0.752 |
| Spur | 119 | 165 | 0.943 | 0.958 | 0.987 | 0.768 |
| Copper | 125 | 148 | 0.985 | 0.973 | 0.993 | 0.911 |
| Pin-hole | 126 | 147 | 0.947 | 0.975 | 0.992 | 0.900 |

**API settings** (conf 0.25, NMS IoU 0.45, all boxes kept): P = 0.939, R = 0.973, F1 = 0.956

**API's single answer per image** (what `/inspect` returns today): the reported defect type is actually present on the board in 150/150 images (100.0%); `no_defect_detected` in 0.

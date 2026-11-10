# Inference benchmark

- **Date:** 2026-11-10
- **CPU:** 12th Gen Intel(R) Core(TM) i7-1260P (16 logical cores)
- **OS / Python:** Windows 11 / Python 3.12.5
- **ONNX Runtime:** 1.29.0, providers: CPUExecutionProvider
- **Model:** `models/pcb_defect_v1.onnx`, 11.70 MB, input 1x3x640x640
- **Dataset:** DeepPCB (Kaggle repackaging, 1,200/150/150 split)
- **Images:** 50 different images from `data/DeepPCB/test/images` (first 50 by filename), each timed once
- **Method:** 5 warm-up runs (not timed), then each image run once, one at a time (batch size 1)

| Measurement | Mean (ms) | Median (ms) | p95 (ms) | Images / sec |
|---|---|---|---|---|
| End-to-end `predict()` (decode + preprocess + model + NMS) | 47.2 | 47.1 | 50.0 | 21.2 |
| Model only (`session.run`) | 36.6 | 36.3 | 39.0 | 27.3 |

These numbers are for this machine only. A small cloud instance (e.g. Render free tier)
has much less CPU and will be slower; network time and database writes are not included.

# PCB Defect Auditor

**Live API:** [https://pcb-defect-auditor.onrender.com/docs](https://pcb-defect-auditor.onrender.com/docs)
*(Hosted on Render's free tier — the first request may take 15–60 seconds to wake the service from sleep. Subsequent requests process in ~3.2–4.2 seconds.)*

An automated optical inspection API for printed circuit boards. A YOLOv8 model, exported to ONNX and served through FastAPI, detects and classifies PCB defects and logs every inspection to a Postgres database (Neon) for later analysis.

Built as a self-directed project to go beyond model training into the full lifecycle: data, training, serving, storage, and deployment.

---

## System Architecture

```
PCB Image Upload
      │
      ▼
FastAPI /inspect endpoint
      │
      ▼
OpenCV preprocessing (resize, normalize, transpose)
      │
      ▼
ONNX Runtime inference (YOLOv8, CPU)
      │
      ▼
NMS + confidence filtering → defect_type, confidence
      │
      ▼
Neon Postgres — inspections table
```

- **Inference:** YOLOv8n, fine-tuned on DeepPCB, exported to ONNX (11.7MB) so the API doesn't need a full PyTorch runtime to serve predictions.
- **Backend:** FastAPI with SQLAlchemy ORM, async endpoints.
- **Storage:** Neon (serverless Postgres) — chosen specifically to sidestep local Docker setup issues during development, and it turned out to be a genuinely good fit for a small project like this.

---

## Core Engineering Decisions

**Server-side payload validation.** Trusting a client-provided HTTP header (like `content-type: image/jpeg`) isn't enough — a client can set that header to anything regardless of what the file actually contains. This architecture decodes the raw byte stream directly with `cv2.imdecode` before it ever reaches the ONNX inference engine. A corrupted byte array or a mislabeled non-image payload (like a PDF sent with a fake image content-type) fails that decode step and is rejected with a clean `400`, rather than crashing further downstream.

**GUI-less containerization.** The deployment strips OS-level display dependencies (e.g. `libgl1`) by using `opencv-python-headless` instead of full OpenCV. Smaller image, faster builds, no display-server dependencies a backend service never needed in the first place.

**Simulation mode for local development without model weights.** `PCBDefectModel` checks whether the ONNX file exists at startup; if it's missing, it falls back to a randomized simulation path instead of crashing, so the API's request/response contract can still be tested. *Note: since `models/pcb_defect_v1.onnx` is committed to this repo (kept intentionally small at 11.7MB), CI and this deployment both run against the real model, not the simulation fallback — the fallback exists for local development on a fresh clone before the file is present, not for hiding weights from CI.*

**Graceful degradation on database failure.** Inference and persistence are decoupled — if the Neon write fails (observed in practice during a free-tier cold start), the API still returns the real inference result to the caller with a `partial_success` status, instead of losing a successful prediction because of an unrelated storage hiccup.

---

## Defect classes

The model detects six PCB fabrication defects, trained on the DeepPCB dataset:

1. **Open** — broken copper trace, causing an open circuit
2. **Short** — unintended bridge between adjacent conductive paths
3. **Mousebite** — small cutouts or jagged edges along a trace
4. **Spur** — unwanted copper protrusion off a trace
5. **Copper** — residual, unetched copper flakes
6. **Pin-hole** — voids in a conductive pad or trace

## Known limitations

- **Out-of-distribution inputs can produce confident false positives.** The model was trained strictly on the DeepPCB distribution and has no "not a PCB" class — feeding it an unrelated image doesn't reliably get rejected. Tested directly: an unrelated non-PCB image returned `"Spur"` at 0.44 confidence rather than a clear rejection.
- **No background/negative class.** There's no dedicated category for "this isn't a defect" — every confident detection gets forced into one of the six known classes.
- **Content-type validation works correctly for non-image files.** A `.pdf` sent to `/inspect` was correctly rejected with a `400`, confirmed directly against the live deployment.
- **Latency on free-tier CPU:** end-to-end inference over the network currently measures ~3.2–4.2 seconds per request.

---

## Evaluation

Trained YOLOv8n for 99 epochs on DeepPCB (early-stopped from a 100-epoch budget, patience=20, best weights from epoch 79). Full training run: `notebooks/train_pcb_model.ipynb`.

**Overall (validation set, 150 images, 984 instances):**

| Metric | Value |
|---|---|
| Precision | 0.974 |
| Recall | 0.952 |
| mAP50 | 0.979 |
| mAP50-95 | 0.791 |

**Per class:**

| Class | mAP50 | mAP50-95 |
|---|---|---|
| Open | 0.988 | 0.731 |
| Short | 0.948 | 0.703 |
| Mousebite | 0.976 | 0.769 |
| Spur | 0.976 | 0.751 |
| Copper | 0.995 | 0.911 |
| Pin-hole | 0.989 | 0.879 |

Full numbers and raw per-epoch log: `docs/training_results/metrics.md` and `results.csv`.

### Confusion matrix

![Confusion Matrix](docs/training_results/confusion_matrix_normalized.png)

The diagonal is strong across every class (0.92–0.99) — the model isn't confusing defect types with each other. The real weak spot is the background column: Mousebite (0.30) and Open (0.27) are the two classes most often triggered by substrate edges or normal board texture that isn't actually a defect. That's a background-vs-defect problem, not a defect-vs-defect problem.

---

## Project structure

```
pcb-defect-auditor/
├── .github/workflows/       # CI pipeline definitions
├── app/
│   ├── api/                 # API route definitions
│   ├── core/                # Config, environment settings
│   ├── db/                  # SQLAlchemy models, database connection
│   ├── ml/                  # ONNX inference session, preprocessing, NMS
│   └── main.py               # FastAPI application entrypoint
├── docs/training_results/    # Confusion matrix, PR curve, training metrics
├── models/
│   └── pcb_defect_v1.onnx    # Exported YOLOv8 weights (committed — see note below)
├── notebooks/                 # Full training notebook
├── scripts/
│   └── convert_labels.py     # Fixes a labeling issue found in the raw dataset
├── sql/
│   ├── schema.sql
│   └── analytics.sql         # Queries for the planned /analytics/summary endpoint
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

**Note on `models/`:** weight files are normally gitignored, but `pcb_defect_v1.onnx` is committed here as an explicit exception (`!models/pcb_defect_v1.onnx` in `.gitignore`), since Render builds directly from this GitHub repo and needs the file present at build time. At 11.7MB, this is a reasonable tradeoff at this project's scale.

---

## A note on the dataset

Getting to a working model took two failed attempts before this one worked, and that's worth leaving in rather than pretending it was smooth:

1. **First attempt** used a Roboflow-hosted export of PKU-Market-PCB. The `data.yaml` class names had somehow been replaced with Roboflow's own boilerplate text instead of real defect labels — only caught after 50 epochs of training produced mAP50 near zero, then visually checking a labeled image against its actual annotations.
2. **Second attempt** with DeepPCB initially failed with "no labels found" — the raw download didn't include YOLO-format annotations at all.
3. **Third attempt**, using a properly YOLO-formatted DeepPCB export, is the one that actually worked — the results above are from that run.

`scripts/convert_labels.py` documents the fix. The broader lesson: a model can train "successfully" — loss decreasing, no errors — against completely wrong labels, and only a direct visual check against ground truth reveals it.

---

## CI/CD & deployment

- **Testing:** GitHub Actions runs the `pytest` suite on Ubuntu, using a mocked in-memory SQLite database and a mocked inference call — no live database or model weights required for the test suite itself to pass.
- **Deployment:** Render pulls this repository directly and builds the Docker image from the committed `Dockerfile`, then runs Uvicorn. This has been verified against a live, successful deploy — see the live link above.

---

## Running it locally

```bash
git clone https://github.com/AarushiSharma1515/pcb-defect-auditor.git
cd pcb-defect-auditor

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Create a `.env` file in the root:
```env
DATABASE_URL=postgresql://<user>:<password>@<neon-host>/<dbname>?sslmode=require
```
(The model path is hardcoded to `models/pcb_defect_v1.onnx` in `inference.py` — no environment variable needed for it.)

Run it:
```bash
uvicorn app.main:app --reload
```
Then open `http://127.0.0.1:8000/docs` for the interactive API.

---

## API

**`GET /`** — health check
```json
{"status": "PCB Defect Auditor is running"}
```

**`GET /inspections`** — list all logged inspections

**`POST /inspect`** — run an inspection
- Form fields: `board_id` (string), `file` (image)
- Response:
```json
{
  "status": "success",
  "board_id": "PANEL-004",
  "requires_human_review": false,
  "telemetry": {
    "defect_type": "Copper",
    "confidence": 0.9142,
    "processing_ms": 4186
  }
}
```

---

---

## Tech stack

Python · FastAPI · SQLAlchemy · PostgreSQL (Neon) · YOLOv8 · ONNX Runtime · OpenCV · pytest · Docker · GitHub Actions
"""
Benchmark the PCB defect ONNX model on CPU.

It uses the exact same PCBDefectModel class the API uses, so the numbers
match what /inspect pays per image (minus network and database time).

Usage (run from the repo root):
    python scripts/benchmark.py --images path/to/DeepPCB/test/images
    python scripts/benchmark.py --images data/DeepPCB/test/images --out docs/benchmark.md \
        --dataset-name "DeepPCB (Kaggle repackaging, 1,200/150/150 split)"
"""
import argparse
import datetime
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Let "from app..." work when this file is run directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from app.ml.inference import PCBDefectModel  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "pcb_defect_v1.onnx"
NUM_IMAGES = 50
WARMUP_RUNS = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def cpu_name() -> str:
    """Human-readable CPU name, e.g. 'Intel(R) Core(TM) i5-1135G7 @ 2.40GHz'."""
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if platform.system() == "Linux":
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def summarize(times_ms: list) -> dict:
    """Mean / median / p95 latency and images per second for a list of timings."""
    arr = np.array(times_ms)
    return {
        "mean": arr.mean(),
        "median": np.median(arr),
        "p95": np.percentile(arr, 95),
        "ips": 1000.0 / arr.mean(),  # one image at a time, so 1000 ms / mean ms
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the ONNX model on CPU")
    parser.add_argument("--images", required=True, help="Folder of held-out test images")
    parser.add_argument("--out", help="Optional path to write the Markdown report to")
    parser.add_argument("--dataset-name", default="unspecified", help="Dataset name shown in the report")
    args = parser.parse_args()

    # 1. Collect images. Use exactly NUM_IMAGES real files; never repeat or fake any.
    folder = Path(args.images)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < NUM_IMAGES:
        sys.exit(f"Need at least {NUM_IMAGES} images in {folder}, found {len(paths)}.")
    paths = paths[:NUM_IMAGES]
    # Read all files into memory first so disk speed is not part of the timing
    images = [p.read_bytes() for p in paths]

    # 2. Load the model exactly like the API does. Refuse to benchmark simulation mode.
    model = PCBDefectModel(model_path=str(MODEL_PATH))
    if not model.use_real_model:
        sys.exit(f"Model not found at {MODEL_PATH}; refusing to benchmark simulation mode.")
    providers = model.session.get_providers()

    # 3. Warm-up: the first runs are slower (memory allocation, caches), so don't time them
    for _ in range(WARMUP_RUNS):
        model.predict(images[0])

    # 4. Time each image two ways:
    #    - end_to_end: model.predict() = decode + preprocess + ONNX run + NMS (what the API does)
    #    - model_only: just session.run() on an already-prepared tensor
    end_to_end_ms, model_only_ms = [], []
    for img_bytes in images:
        start = time.perf_counter()
        model.predict(img_bytes)
        end_to_end_ms.append((time.perf_counter() - start) * 1000)

        tensor = model.preprocess(img_bytes)
        start = time.perf_counter()
        model.session.run(None, {model.input_name: tensor})
        model_only_ms.append((time.perf_counter() - start) * 1000)

    e2e = summarize(end_to_end_ms)
    raw = summarize(model_only_ms)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    # 5. Build a Markdown report (printed, and optionally saved)
    report = f"""# Inference benchmark

- **Date:** {datetime.date.today().isoformat()}
- **CPU:** {cpu_name()} ({os.cpu_count()} logical cores)
- **OS / Python:** {platform.system()} {platform.release()} / Python {platform.python_version()}
- **ONNX Runtime:** {ort.__version__}, providers: {", ".join(providers)}
- **Model:** `models/pcb_defect_v1.onnx`, {size_mb:.2f} MB, input 1x3x640x640
- **Dataset:** {args.dataset_name}
- **Images:** {NUM_IMAGES} different images from `{folder.as_posix()}` (first {NUM_IMAGES} by filename), each timed once
- **Method:** {WARMUP_RUNS} warm-up runs (not timed), then each image run once, one at a time (batch size 1)

| Measurement | Mean (ms) | Median (ms) | p95 (ms) | Images / sec |
|---|---|---|---|---|
| End-to-end `predict()` (decode + preprocess + model + NMS) | {e2e['mean']:.1f} | {e2e['median']:.1f} | {e2e['p95']:.1f} | {e2e['ips']:.1f} |
| Model only (`session.run`) | {raw['mean']:.1f} | {raw['median']:.1f} | {raw['p95']:.1f} | {raw['ips']:.1f} |

These numbers are for this machine only. A small cloud instance (e.g. Render free tier)
has much less CPU and will be slower; network time and database writes are not included.
"""
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

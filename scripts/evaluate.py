"""
Evaluate the ONNX model on labelled DeepPCB splits: precision, recall, mAP50, mAP50-95.

Each split folder must contain images/ and labels/, where each label file has one
box per line in raw DeepPCB format:  x1 y1 x2 y2 type   (pixels, type 1-6).

Two settings are reported for every split:
  - "Ultralytics-style": conf 0.001, NMS IoU 0.7, per-class NMS, multi-label.
    This mirrors how `yolo val` scores a model, so it is comparable to the
    mAP50 = 0.979 in docs/training_results/metrics.md.
  - "API settings": conf 0.25, NMS IoU 0.45, one class per box, class-agnostic NMS.
    This is what the deployed API's thresholds give.

Usage (from the repo root):
    python scripts/evaluate.py --data data/DeepPCB --splits test valid --out docs/evaluation.md \
        --dataset-name "DeepPCB (Kaggle repackaging, 1,200/150/150 split)"
"""
import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np

# Let "from app..." work when this file is run directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from app.ml.inference import PCBDefectModel  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "pcb_defect_v1.onnx"
MODEL_SIZE = 640                                # model input is 640x640
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)     # 0.50, 0.55, ... 0.95 (for mAP50-95)
MAX_DETECTIONS = 300                            # same cap Ultralytics uses
EPS = 1e-16

SETTINGS = {
    "ultralytics": {"conf": 0.001, "iou": 0.7, "multi_label": True, "agnostic": False},
    "api": {"conf": 0.25, "iou": 0.45, "multi_label": False, "agnostic": True},
}


# ---------------------------------------------------------------- data loading
def load_labels(path: Path):
    """Read 'x1 y1 x2 y2 type' lines -> boxes (N, 4) in pixels and classes (N,) as 0-5."""
    boxes, classes = [], []
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            continue
        x1, y1, x2, y2, t = (float(v) for v in parts[:5])
        boxes.append([x1, y1, x2, y2])
        classes.append(int(t) - 1)  # DeepPCB types are 1-6; model classes are 0-5
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(classes, dtype=int)


# ---------------------------------------------------------------- inference
def detect(model, image_bytes, height, width, conf, iou, multi_label, agnostic):
    """Run the ONNX model and return ALL detections after NMS: boxes (x1,y1,x2,y2), scores, classes."""
    tensor = model.preprocess(image_bytes)                       # same preprocessing as the API
    output = model.session.run(None, {model.input_name: tensor})[0]
    preds = output[0].T                                          # (8400, 4 + num_classes)
    xywh, class_scores = preds[:, :4], preds[:, 4:]

    if multi_label:
        # A box may count for several classes if each class score passes the threshold
        rows, classes = np.nonzero(class_scores > conf)
        scores = class_scores[rows, classes]
        xywh = xywh[rows]
    else:
        # One class per box: the highest-scoring one (this is what the API does)
        classes = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        keep = scores > conf
        xywh, scores, classes = xywh[keep], scores[keep], classes[keep]

    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # Model boxes are (center x, center y, w, h) in 640x640 space -> corners in original pixels
    sx, sy = width / MODEL_SIZE, height / MODEL_SIZE
    x1 = (xywh[:, 0] - xywh[:, 2] / 2) * sx
    y1 = (xywh[:, 1] - xywh[:, 3] / 2) * sy
    x2 = (xywh[:, 0] + xywh[:, 2] / 2) * sx
    y2 = (xywh[:, 1] + xywh[:, 3] / 2) * sy
    corners = np.stack([x1, y1, x2, y2], axis=1)

    # OpenCV NMS expects [x, y, width, height], NOT corners
    rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    if agnostic:
        keep = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf, iou)
    else:
        keep = cv2.dnn.NMSBoxesBatched(rects, scores.tolist(), classes.tolist(), conf, iou)
    keep = np.array(keep, dtype=int).flatten()
    keep = keep[np.argsort(-scores[keep])][:MAX_DETECTIONS]     # highest score first, max 300
    return corners[keep], scores[keep], classes[keep]


# ---------------------------------------------------------------- matching
def box_iou(a, b):
    """IoU between every box in a (N, 4) and every box in b (M, 4) -> (N, M)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + EPS)


def match(pred_boxes, pred_classes, gt_boxes, gt_classes):
    """For each prediction and IoU threshold, is it a true positive? -> (num_preds, 10) bool.

    Same idea as Ultralytics: a prediction can match one ground-truth box of the same
    class with IoU >= threshold; each ground-truth box can be matched only once;
    the highest-IoU pairs are matched first.
    """
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou(gt_boxes, pred_boxes)                         # (num_gt, num_preds)
    iou[gt_classes[:, None] != pred_classes[None, :]] = 0       # only same-class pairs count
    for t, threshold in enumerate(IOU_THRESHOLDS):
        gt_idx, pred_idx = np.nonzero(iou >= threshold)
        if len(gt_idx) == 0:
            continue
        order = np.argsort(-iou[gt_idx, pred_idx])              # best overlaps first
        gt_idx, pred_idx = gt_idx[order], pred_idx[order]
        _, first = np.unique(pred_idx, return_index=True)       # each prediction used once
        gt_idx, pred_idx = gt_idx[first], pred_idx[first]
        order = np.argsort(-iou[gt_idx, pred_idx])
        gt_idx, pred_idx = gt_idx[order], pred_idx[order]
        _, first = np.unique(gt_idx, return_index=True)         # each ground truth used once
        correct[pred_idx[first], t] = True
    return correct


# ---------------------------------------------------------------- metrics
def average_precision(recall, precision):
    """Area under the precision-recall curve, 101-point interpolation (as Ultralytics/COCO)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))        # make precision never increase
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x)


def smooth(y, fraction):
    """Moving average (Ultralytics smooths the F1 curve before picking the best threshold)."""
    n = round(len(y) * fraction * 2) // 2 + 1
    pad = np.ones(n // 2)
    padded = np.concatenate((pad * y[0], y, pad * y[-1]))
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def compute_metrics(correct, scores, pred_classes, gt_classes, num_classes, best_f1):
    """Per-class precision, recall, AP50 and AP50-95.

    best_f1=True:  P and R at the confidence where mean F1 is highest (how `yolo val` reports them)
    best_f1=False: P and R using every detection that passed the confidence threshold
    """
    order = np.argsort(-scores)
    correct, scores, pred_classes = correct[order], scores[order], pred_classes[order]
    conf_grid = np.linspace(0, 1, 1000)
    ap = np.zeros((num_classes, len(IOU_THRESHOLDS)))
    p_curve = np.zeros((num_classes, 1000))
    r_curve = np.zeros((num_classes, 1000))
    p_final = np.zeros(num_classes)
    r_final = np.zeros(num_classes)

    for c in range(num_classes):
        mask = pred_classes == c
        n_gt = int((gt_classes == c).sum())
        if mask.sum() == 0 or n_gt == 0:
            continue
        tp_cum = correct[mask].cumsum(axis=0)
        fp_cum = (~correct[mask]).cumsum(axis=0)
        recall = tp_cum / (n_gt + EPS)
        precision = tp_cum / (tp_cum + fp_cum)
        for t in range(len(IOU_THRESHOLDS)):
            ap[c, t] = average_precision(recall[:, t], precision[:, t])
        # Precision/recall as a function of confidence threshold (at IoU 0.5)
        r_curve[c] = np.interp(-conf_grid, -scores[mask], recall[:, 0], left=0)
        p_curve[c] = np.interp(-conf_grid, -scores[mask], precision[:, 0], left=1)
        p_final[c], r_final[c] = precision[-1, 0], recall[-1, 0]

    if best_f1:
        f1 = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
        best = smooth(f1.mean(axis=0), 0.1).argmax()
        p, r = p_curve[:, best], r_curve[:, best]
    else:
        p, r = p_final, r_final
    return {"p": p, "r": r, "ap50": ap[:, 0], "ap50_95": ap.mean(axis=1)}


# ---------------------------------------------------------------- per split
def evaluate_split(model, split_dir: Path):
    images = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    names = model.defect_classes
    num_classes = len(names)
    results = {}
    gt_classes_all = []
    images_with_class = np.zeros(num_classes, dtype=int)
    top1_hits, top1_none = 0, 0
    collected = {mode: {"correct": [], "scores": [], "classes": []} for mode in SETTINGS}

    for image_path in images:
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            sys.exit(f"Missing label for {image_path.name}")
        gt_boxes, gt_classes = load_labels(label_path)
        gt_classes_all.append(gt_classes)
        images_with_class[np.unique(gt_classes)] += 1

        image_bytes = image_path.read_bytes()
        height, width = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR).shape[:2]

        for mode, s in SETTINGS.items():
            boxes, scores, classes = detect(model, image_bytes, height, width, **s)
            collected[mode]["correct"].append(match(boxes, classes, gt_boxes, gt_classes))
            collected[mode]["scores"].append(scores)
            collected[mode]["classes"].append(classes)

        # The real API path: one answer per image. Is it a defect type that is really on the board?
        answer = model.predict(image_bytes)["defect_type"]
        if answer == "no_defect_detected":
            top1_none += 1
        elif answer in {names[c] for c in gt_classes}:
            top1_hits += 1

    gt_classes_all = np.concatenate(gt_classes_all)
    for mode in SETTINGS:
        col = collected[mode]
        results[mode] = compute_metrics(
            np.concatenate(col["correct"]), np.concatenate(col["scores"]), np.concatenate(col["classes"]),
            gt_classes_all, num_classes, best_f1=(mode == "ultralytics"),
        )
    instances = np.bincount(gt_classes_all, minlength=num_classes)
    return {
        "num_images": len(images), "instances": instances, "images_with_class": images_with_class,
        "results": results, "top1_hits": top1_hits, "top1_none": top1_none,
    }


# ---------------------------------------------------------------- report
def split_report(split, folder, r, names):
    u, a = r["results"]["ultralytics"], r["results"]["api"]
    lines = [
        f"## {split} split (`{folder}`)",
        "",
        f"{r['num_images']} images, {r['instances'].sum()} labelled defects.",
        "",
        "**Ultralytics-style** (conf 0.001, NMS IoU 0.7; P and R at the best-F1 confidence):",
        "",
        "| Class | Images | Instances | P | R | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|---|",
        f"| **all** | {r['num_images']} | {r['instances'].sum()} | {u['p'].mean():.3f} | {u['r'].mean():.3f} "
        f"| {u['ap50'].mean():.3f} | {u['ap50_95'].mean():.3f} |",
    ]
    for c, name in enumerate(names):
        lines.append(
            f"| {name} | {r['images_with_class'][c]} | {r['instances'][c]} | {u['p'][c]:.3f} | {u['r'][c]:.3f} "
            f"| {u['ap50'][c]:.3f} | {u['ap50_95'][c]:.3f} |"
        )
    p, rec = a["p"].mean(), a["r"].mean()
    lines += [
        "",
        f"**API settings** (conf 0.25, NMS IoU 0.45, all boxes kept): "
        f"P = {p:.3f}, R = {rec:.3f}, F1 = {2 * p * rec / (p + rec + EPS):.3f}",
        "",
        f"**API's single answer per image** (what `/inspect` returns today): the reported defect type is "
        f"actually present on the board in {r['top1_hits']}/{r['num_images']} images "
        f"({100 * r['top1_hits'] / r['num_images']:.1f}%); `no_defect_detected` in {r['top1_none']}.",
        "",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ONNX model on labelled DeepPCB splits")
    parser.add_argument("--data", required=True, help="Folder containing the split folders")
    parser.add_argument("--splits", nargs="+", default=["test", "valid"], help="Split folder names")
    parser.add_argument("--out", help="Optional path to write the Markdown report to")
    parser.add_argument("--dataset-name", default="unspecified", help="Dataset name shown in the report")
    args = parser.parse_args()

    model = PCBDefectModel(model_path=str(MODEL_PATH))
    if not model.use_real_model:
        sys.exit(f"Model not found at {MODEL_PATH}; refusing to evaluate simulation mode.")

    lines = [
        "# Model evaluation",
        "",
        f"- **Date:** {datetime.date.today().isoformat()}",
        "- **Model:** `models/pcb_defect_v1.onnx` (ONNX Runtime, CPU), same preprocessing as the API",
        f"- **Dataset:** {args.dataset_name}",
        "- **Matching:** a detection is correct if it has the right class and IoU >= 0.5 with an unmatched "
        "labelled box (IoU 0.50-0.95 for mAP50-95); AP uses 101-point interpolation",
        "",
    ]
    for split in args.splits:
        folder = Path(args.data) / split
        print(f"Evaluating {folder} ...", flush=True)
        lines += split_report(split, folder.as_posix(), evaluate_split(model, folder), model.defect_classes)

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

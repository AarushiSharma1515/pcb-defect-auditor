import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Find the model relative to THIS file (app/ml/inference.py -> up 2 folders = repo root),
# not relative to the folder the app was started from, so it works from any directory.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pcb_defect_v1.onnx"


class PCBDefectModel:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)

        # Must exactly match the class order used in data.yaml during training
        self.defect_classes = ["Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"]

        # No model, no service: stop with a clear error instead of returning made-up predictions
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. The API cannot run without it; "
                f"make sure models/pcb_defect_v1.onnx is in the repo and in the Docker image."
            )
        print(f"Initializing ONNX Runtime with {self.model_path}...")
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        # 1. Decode raw bytes into a BGR image array directly in memory
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image — file may be corrupted or not a valid image")

        # 2. Resize to standard YOLO dimension
        img_resized = cv2.resize(img, (640, 640))

        # 3. Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 4. Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 5. Transpose from (H, W, C) to (C, H, W) for PyTorch/ONNX standard
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # 6. Add batch dimension to create [1, 3, 640, 640]
        input_tensor = np.expand_dims(img_transposed, axis=0)

        return input_tensor

    def detect(self, outputs, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Parses raw YOLOv8 ONNX output into EVERY detection that survives NMS.
        Output shape from export: (1, num_classes + 4, num_anchors) -> e.g. (1, 10, 8400)
        First 4 rows are box coords (cx, cy, w, h); remaining rows are per-class scores.

        Returns a list of (class_id, confidence, [x1, y1, x2, y2]) with the box in the
        model's 640x640 input space, highest confidence first. Empty list if nothing found.
        """
        predictions = np.squeeze(outputs[0]).T  # -> (num_anchors, num_classes + 4)

        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter low-confidence detections
        mask = confidences > conf_threshold
        boxes, class_ids, confidences = boxes[mask], class_ids[mask], confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert center-format (cx, cy, w, h) to corners (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # OpenCV's NMS expects each box as [x, y, width, height], NOT corners. Passing
        # corners makes it read x2/y2 as a width/height, so every box looks far bigger
        # than it is and neighbouring defects get merged into one.
        nms_rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            nms_rects.tolist(), confidences.tolist(), conf_threshold, iou_threshold
        )

        # The return shape varies by OpenCV version ((N,) or (N, 1)); flatten handles both.
        # NMSBoxes returns indices sorted by score, highest first.
        indices = np.array(indices, dtype=int).flatten()

        return [
            (int(class_ids[i]), float(confidences[i]),
             [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])])
            for i in indices
        ]

    def postprocess(self, outputs, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """The single best (defect_type, confidence) pair, or (None, 0.0) if nothing was found."""
        detections = self.detect(outputs, conf_threshold, iou_threshold)
        if not detections:
            return None, 0.0
        best_class_id, best_confidence, _ = detections[0]
        return self.defect_classes[best_class_id], best_confidence

    def predict(self, image_bytes: bytes) -> dict:
        start_time = time.perf_counter()
        input_tensor = self.preprocess(image_bytes)   # raises ValueError if the image can't be decoded
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predicted_defect, confidence = self.postprocess(outputs)

        if predicted_defect is None:
            predicted_defect = "no_defect_detected"
            confidence = 0.0
        end_time = time.perf_counter()

        return {
            "defect_type": predicted_defect,
            "confidence": round(confidence, 4),
            "processing_ms": int((end_time - start_time) * 1000)
        }
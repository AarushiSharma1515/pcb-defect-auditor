import os
import time
import random
import cv2
import numpy as np
import onnxruntime as ort


class PCBDefectModel:
    def __init__(self, model_path: str = "models/pcb_defect_v1.onnx"):
        self.model_path = model_path

        # Must exactly match the class order used in data.yaml during training
        self.defect_classes = ["Open", "Short", "Mousebite", "Spur", "Copper", "Pin-hole"]

        # Check if the real model exists yet
        self.use_real_model = os.path.exists(model_path)
        if self.use_real_model:
            print(f"Initializing ONNX Runtime with {model_path}...")
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
        else:
            print(f"Warning: {model_path} not found. Running in simulation mode.")

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

    def postprocess(self, outputs, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Parses raw YOLOv8 ONNX output into a single best (defect_type, confidence) pair.
        Output shape from export: (1, num_classes + 4, num_anchors) -> e.g. (1, 10, 8400)
        First 4 rows are box coords (cx, cy, w, h); remaining rows are per-class scores.
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
            return None, 0.0

        # Convert center-format (cx, cy, w, h) to corner-format (x1, y1, x2, y2) for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        nms_boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold
        )

        if len(indices) == 0:
            return None, 0.0

        # cv2.dnn.NMSBoxes return shape varies by OpenCV version — handle both
        best_idx = indices[0] if isinstance(indices[0], (int, np.integer)) else indices[0][0]

        best_class_id = int(class_ids[best_idx])
        best_confidence = float(confidences[best_idx])

        return self.defect_classes[best_class_id], best_confidence

    def predict(self, image_bytes: bytes) -> dict:
        start_time = time.perf_counter()

        if self.use_real_model:
            # Real ONNX Execution Pipeline
            input_tensor = self.preprocess(image_bytes)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            predicted_defect, confidence = self.postprocess(outputs)

            if predicted_defect is None:
                predicted_defect = "no_defect_detected"
                confidence = 0.0
        else:
            # Simulation Pipeline (Fallback — used only if weights file is missing)
            time.sleep(random.uniform(0.01, 0.04))
            predicted_defect = random.choices(
                self.defect_classes, weights=[1] * len(self.defect_classes)
            )[0]
            confidence = round(random.uniform(0.85, 0.99), 4)

        end_time = time.perf_counter()

        return {
            "defect_type": predicted_defect,
            "confidence": round(confidence, 4),
            "processing_ms": int((end_time - start_time) * 1000)
        }
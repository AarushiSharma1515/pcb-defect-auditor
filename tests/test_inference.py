"""Tests for turning raw model output into detections (app/ml/inference.py).

These feed hand-made model outputs to the post-processing code, so they test our box
maths and NMS exactly, independent of what the real model happens to predict.
"""
import numpy as np
import pytest

from app.ml.inference import PCBDefectModel

NUM_CLASSES = 6
NUM_ANCHORS = 8400
SHORT = 1  # index of "Short" in PCBDefectModel.defect_classes


@pytest.fixture(scope="module")
def model():
    return PCBDefectModel()


def fake_output(boxes):
    """Raw YOLOv8 output (1, 4 + classes, anchors) holding the given boxes.

    Each box is (cx, cy, w, h, class_id, score) in the model's 640x640 space;
    every other anchor has score 0, so it is filtered out.
    """
    out = np.zeros((1, 4 + NUM_CLASSES, NUM_ANCHORS), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(boxes):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + class_id, i] = score
    return [out]


def test_two_separate_defects_stay_separate(model):
    # Two 20x20 defects with a 20 px gap between them: no overlap at all.
    # (Before the fix, OpenCV read the corners as x, y, WIDTH, HEIGHT, saw two
    # ~300 px boxes overlapping by IoU ~0.69, and deleted the weaker one.)
    outputs = fake_output([
        (300, 300, 20, 20, SHORT, 0.90),
        (340, 300, 20, 20, SHORT, 0.80),
    ])
    detections = model.detect(outputs)

    assert len(detections) == 2
    assert [round(conf, 2) for _, conf, _ in detections] == [0.90, 0.80]  # best first
    assert detections[0][2] == pytest.approx([290, 290, 310, 310])
    assert detections[1][2] == pytest.approx([330, 290, 350, 310])


def test_duplicate_boxes_for_one_defect_are_still_merged(model):
    # The same defect found twice, shifted by 2 px (IoU ~0.82 > 0.45): NMS keeps only the best
    outputs = fake_output([
        (300, 300, 20, 20, SHORT, 0.70),
        (302, 300, 20, 20, SHORT, 0.95),
    ])
    detections = model.detect(outputs)

    assert len(detections) == 1
    assert detections[0][1] == pytest.approx(0.95)


def test_postprocess_still_returns_the_top_defect(model):
    outputs = fake_output([
        (300, 300, 20, 20, SHORT, 0.60),
        (100, 100, 20, 20, 0, 0.90),  # "Open", higher score
    ])
    defect, confidence = model.postprocess(outputs)
    assert defect == "Open"
    assert confidence == pytest.approx(0.90)


def test_nothing_above_threshold_gives_no_detections(model):
    outputs = fake_output([(300, 300, 20, 20, SHORT, 0.10)])  # below the 0.25 threshold
    assert model.detect(outputs) == []
    assert model.postprocess(outputs) == (None, 0.0)

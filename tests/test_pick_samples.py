"""Tests for scripts/pick_samples.py, on a tiny made-up dataset (never the real data/ folder)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pick_samples.py"
spec = importlib.util.spec_from_file_location("pick_samples", SCRIPT)
pick_samples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick_samples)


def make_split(root: Path, labels: dict) -> Path:
    """A split folder with images/ and labels/. labels = {image stem: [type, type, ...]}."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for stem, types in labels.items():
        (root / "images" / f"{stem}.jpg").write_bytes(f"image {stem}".encode())
        (root / "labels" / f"{stem}.txt").write_text(
            "".join(f"10 10 20 20 {t}\n" for t in types)
        )
    return root


# Every class appears in a "mixed" image; each class also has one cleaner image
DATASET = {
    "mixed":     [1, 2, 3, 4, 5, 6],
    "open_only": [1, 1],
    "short_2":   [2, 1],          # one other box
    "short_1":   [2],             # cleanest Short image -> chosen
    "mouse":     [3, 3, 3],
    "mouse_few": [3],             # also clean, but fewer Mousebite boxes -> not chosen
    "spur":      [4],
    "copper":    [5, 6],
    "pinhole":   [6],
}


def test_picks_one_clean_image_per_class(tmp_path):
    chosen = pick_samples.pick(make_split(tmp_path / "valid", DATASET))

    assert [(t, image.stem) for t, image, _ in chosen] == [
        (1, "open_only"), (2, "short_1"), (3, "mouse"), (4, "spur"), (5, "copper"), (6, "pinhole"),
    ]


def test_each_class_gets_a_different_image(tmp_path):
    # "both" is the cleanest image for BOTH Open and Short, but may only be used once
    data = {"both": [1, 2], "open2": [1, 3, 3], "short2": [2, 4], "m": [3], "s": [4], "c": [5], "p": [6]}
    chosen = pick_samples.pick(make_split(tmp_path / "valid", data))
    stems = [image.stem for _, image, _ in chosen]
    assert len(set(stems)) == 6
    assert stems[:2] == ["both", "short2"]


def test_copies_images_and_writes_readme(tmp_path):
    src = make_split(tmp_path / "valid", DATASET)
    dest = tmp_path / "samples"

    assert pick_samples.main(["--src", str(src), "--dest", str(dest)]) == 0

    images = sorted(p.name for p in dest.glob("*.jpg"))
    assert images == [
        "1_open_open_only.jpg", "2_short_short_1.jpg", "3_mousebite_mouse.jpg",
        "4_spur_spur.jpg", "5_copper_copper.jpg", "6_pin-hole_pinhole.jpg",
    ]
    assert (dest / "2_short_short_1.jpg").read_bytes() == b"image short_1"   # an exact copy

    # The README is fixed text: it never lists the picked files, so re-running never changes it
    assert (dest / "README.md").read_text(encoding="utf-8") == pick_samples.README_TEXT
    assert "short_1" not in pick_samples.README_TEXT


def test_readme_has_the_exact_credit_line():
    readme = pick_samples.README_TEXT
    assert pick_samples.CREDIT_LINE in readme
    assert "https://github.com/tangsanli5201/DeepPCB" in readme
    assert "'On-line PCB Defect Detector On A New PCB Defect Dataset'" in readme
    assert "may only be used for research purposes" in readme
    assert "The current version of this repository contains no dataset images." in readme
    assert "redistributed" not in readme       # the images remain in git history, so don't claim that
    assert "intentionally empty in the repo" in readme
    assert "for demos only" in readme
    assert "et al" not in readme                # credit by repo URL and paper title, not author citation


def test_main_readme_has_the_same_credit_line():
    main_readme = (SCRIPT.parents[1] / "README.md").read_text(encoding="utf-8")
    assert pick_samples.CREDIT_LINE in main_readme


def test_committed_readme_matches_what_the_script_writes():
    committed = SCRIPT.parents[1] / "samples" / "README.md"
    assert committed.read_text(encoding="utf-8") == pick_samples.README_TEXT


def test_dry_run_copies_nothing(tmp_path):
    src = make_split(tmp_path / "valid", DATASET)
    dest = tmp_path / "samples"
    assert pick_samples.main(["--src", str(src), "--dest", str(dest), "--dry-run"]) == 0
    assert not dest.exists()


def test_a_missing_class_stops_before_copying_anything(tmp_path):
    no_pinhole = {k: [t for t in v if t != 6] for k, v in DATASET.items()}
    src = make_split(tmp_path / "valid", no_pinhole)
    dest = tmp_path / "samples"
    with pytest.raises(SystemExit, match="Pin-hole"):
        pick_samples.main(["--src", str(src), "--dest", str(dest)])
    assert not dest.exists()


def test_sample_names_are_accepted_by_the_samples_endpoint(tmp_path):
    # /samples/{name} rejects names containing "..", "/" or "\"
    chosen = pick_samples.pick(make_split(tmp_path / "valid", DATASET))
    for t, image, _ in chosen:
        name = pick_samples.sample_name(t, image)
        assert ".." not in name and "/" not in name and "\\" not in name

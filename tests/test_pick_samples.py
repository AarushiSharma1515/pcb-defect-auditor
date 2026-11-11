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


def test_copies_images_and_writes_readme(tmp_path, capsys):
    src = make_split(tmp_path / "valid", DATASET)
    dest = tmp_path / "samples"

    assert pick_samples.main(["--src", str(src), "--dest", str(dest), "--licence", "Test licence 1.0"]) == 0

    images = sorted(p.name for p in dest.glob("*.jpg"))
    assert images == [
        "1_open_open_only.jpg", "2_short_short_1.jpg", "3_mousebite_mouse.jpg",
        "4_spur_spur.jpg", "5_copper_copper.jpg", "6_pin-hole_pinhole.jpg",
    ]
    assert (dest / "2_short_short_1.jpg").read_bytes() == b"image short_1"   # an exact copy

    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "**Source:** DeepPCB" in readme
    assert "**Licence:** Test licence 1.0" in readme
    assert "`short_1.jpg`" in readme                  # names the original file
    assert "WARNING" not in capsys.readouterr().out


def test_warns_when_licence_is_still_a_placeholder(tmp_path, capsys):
    src = make_split(tmp_path / "valid", DATASET)
    pick_samples.main(["--src", str(src), "--dest", str(tmp_path / "samples")])
    assert pick_samples.LICENCE_PLACEHOLDER in (tmp_path / "samples" / "README.md").read_text(encoding="utf-8")
    assert "WARNING" in capsys.readouterr().out


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

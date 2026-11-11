"""
Copy ONE demo image per defect class from the DeepPCB validation split into samples/,
for a LOCAL demo. GET /samples then serves them.

The copies never leave your machine: .gitignore and .dockerignore exclude samples/*.jpg,
*.jpeg and *.png, because the DeepPCB repository states the dataset may only be used for
research purposes, and the current version of this repository contains no dataset images. The script also
(re)writes samples/README.md, which IS tracked; its text is fixed, so re-running the
script never changes it. Which images were picked is printed, not written to the README.

How an image is chosen for each class (using the label files, not the model):
  1. its label file contains at least one box of that class
  2. fewest boxes of OTHER classes (ideally the image shows only that defect)
  3. then the most boxes of that class
  4. then file name, so the choice is the same on every run
Each class gets a different image. If any class can't be filled, nothing is copied.

Label files are raw DeepPCB format, one box per line:  x1 y1 x2 y2 type   (type 1-6).

Usage (from the repo root):
    python scripts/pick_samples.py --dry-run     # show the choices, copy nothing
    python scripts/pick_samples.py
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

# Let "from app..." work when this file is run directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from app.ml.inference import DEFECT_CLASSES  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "data" / "DeepPCB" / "valid"
DEFAULT_DEST = REPO_ROOT / "samples"
IMAGE_EXTENSIONS = (".jpg", ".png")

CREDIT_LINE = (
    'Trained and evaluated on the DeepPCB dataset (https://github.com/tangsanli5201/DeepPCB), '
    'associated with the paper "On-line PCB Defect Detector On A New PCB Defect Dataset", '
    'using the Kaggle repackaging "pcb-defect-detection-dataset" by liuxiaolong1. '
    'The DeepPCB repository states that the dataset may only be used for research purposes; '
    'this project is a non-commercial portfolio demonstration. '
    'The current version of this repository contains no dataset images.'
)

README_TEXT = f"""# Sample images

This folder is intentionally empty in the current version of the repository.
`.gitignore` and `.dockerignore` exclude `samples/*.jpg`, `samples/*.jpeg` and
`samples/*.png`, so images copied here are not committed and not put in the Docker image.

For a local demo, run this from the repo root:

```
python scripts/pick_samples.py --dry-run   # see which images it would pick
python scripts/pick_samples.py             # copy them here
```

It copies six images from `data/DeepPCB/valid/images` into this folder, one per defect
class (Open, Short, Mousebite, Spur, Copper, Pin-hole), chosen from the label files.
`GET /samples` then lists them and `GET /samples/{{name}}` serves them. The copies stay
on your machine.

## Dataset credit

{CREDIT_LINE}
"""


def read_label_types(path: Path) -> Counter:
    """Count the boxes of each DeepPCB type (1-6) in one label file."""
    counts = Counter()
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 5:
            counts[int(float(parts[4]))] += 1
    return counts


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def pick(src: Path):
    """Return [(type 1-6, image path, box counts of that image)], one per class, in class order."""
    images_dir, labels_dir = src / "images", src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist.")

    labelled = []  # (image path, counts) for every image that has a label file with boxes
    for label_file in sorted(labels_dir.glob("*.txt")):
        counts = read_label_types(label_file)
        image = find_image(images_dir, label_file.stem)
        if counts and image is not None:
            labelled.append((image, counts))

    chosen, used = [], set()
    for defect_type in range(1, len(DEFECT_CLASSES) + 1):
        candidates = [
            (sum(c.values()) - c[defect_type], -c[defect_type], image.name, image, c)
            for image, c in labelled
            if c[defect_type] > 0 and image.name not in used
        ]
        if not candidates:
            raise SystemExit(
                f"No unused image in {labels_dir} contains class {defect_type} "
                f"({DEFECT_CLASSES[defect_type - 1]}). Nothing was copied."
            )
        *_, image, counts = min(candidates)
        used.add(image.name)
        chosen.append((defect_type, image, counts))
    return chosen


def sample_name(defect_type: int, image: Path) -> str:
    """e.g. 1_open_01_PCB__10.jpg: the number keeps the classes in order in /samples."""
    return f"{defect_type}_{DEFECT_CLASSES[defect_type - 1].lower()}_{image.name}"


def describe(counts: Counter) -> str:
    return ", ".join(f"{counts[t]} {DEFECT_CLASSES[t - 1]}" for t in sorted(counts))


def write_readme(dest: Path):
    # newline="\n" so the file is byte-for-byte the same on Windows and Linux
    with open(dest / "README.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(README_TEXT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="split folder with images/ and labels/")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where to copy the samples")
    parser.add_argument("--dry-run", action="store_true", help="print the choices, copy nothing")
    args = parser.parse_args(argv)

    chosen = pick(args.src)
    for t, image, counts in chosen:
        print(f"{DEFECT_CLASSES[t - 1]:<10} {image.name:<24} -> {sample_name(t, image)}   ({describe(counts)})")
    if args.dry_run:
        print("Dry run: nothing copied.")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for t, image, _ in chosen:
        shutil.copy2(image, args.dest / sample_name(t, image))
    write_readme(args.dest)
    print(f"Copied {len(chosen)} images into {args.dest} (ignored by git and Docker; keep them local).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

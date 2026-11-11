"""
Copy ONE demo image per defect class from the DeepPCB validation split into samples/,
and write samples/README.md saying where they came from. GET /samples then serves them.

How an image is chosen for each class (using the label files, not the model):
  1. its label file contains at least one box of that class
  2. fewest boxes of OTHER classes (ideally the image shows only that defect)
  3. then the most boxes of that class
  4. then file name, so the choice is the same on every run
Each class gets a different image. If any class can't be filled, nothing is copied.

Label files are raw DeepPCB format, one box per line:  x1 y1 x2 y2 type   (type 1-6).

Usage (from the repo root):
    python scripts/pick_samples.py --dry-run                 # show the choices, copy nothing
    python scripts/pick_samples.py --licence "<licence line>"
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

SOURCE_LINE = (
    "DeepPCB dataset, validation split (`data/DeepPCB/valid`) of the Kaggle repackaging "
    "`liuxiaolong1/pcb-defect-detection-dataset` that this model was evaluated on. "
    "Original dataset: https://github.com/tangsanli5201/DeepPCB"
)
# Paste the dataset's licence line here, or pass it with --licence "...".
LICENCE_PLACEHOLDER = "<PASTE THE LICENCE LINE HERE>"
LICENCE_LINE = LICENCE_PLACEHOLDER


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


def write_readme(dest: Path, chosen, licence: str):
    rows = "\n".join(
        f"| `{sample_name(t, image)}` | {DEFECT_CLASSES[t - 1]} | `{image.name}` | {describe(counts)} |"
        for t, image, counts in chosen
    )
    (dest / "README.md").write_text(
        "# Sample images\n\n"
        "One demo image per defect class, served by `GET /samples` and `GET /samples/{name}`.\n"
        "Chosen by `scripts/pick_samples.py` from the label files: the image with the fewest\n"
        "other defect types labelled in it.\n\n"
        f"**Source:** {SOURCE_LINE}\n\n"
        f"**Licence:** {licence}\n\n"
        "| Sample | Class | Original file | Labelled defects in the image |\n"
        "|---|---|---|---|\n"
        f"{rows}\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="split folder with images/ and labels/")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="where to copy the samples")
    parser.add_argument("--licence", default=LICENCE_LINE, help="licence line for samples/README.md")
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
    write_readme(args.dest, chosen, args.licence)
    print(f"Copied {len(chosen)} images and wrote {args.dest / 'README.md'}")
    if args.licence == LICENCE_PLACEHOLDER:
        print("WARNING: the licence line is still a placeholder. Edit samples/README.md, "
              "or re-run with --licence \"...\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Sample images

This folder is intentionally empty in the repo. `scripts/pick_samples.py` copies six
images into it locally, for demos only. `.gitignore` and `.dockerignore` exclude
`samples/*.jpg`, `samples/*.jpeg` and `samples/*.png`, so images copied here are not
committed and not put in the Docker image.

To fill it, run this from the repo root:

```
python scripts/pick_samples.py --dry-run   # see which images it would pick
python scripts/pick_samples.py             # copy them here
```

It copies six images from `data/DeepPCB/valid/images` into this folder, one per defect
class (Open, Short, Mousebite, Spur, Copper, Pin-hole), chosen from the label files.
`GET /samples` then lists them and `GET /samples/{name}` serves them. The copies stay
on your machine.

## Dataset credit

Trained and evaluated on the DeepPCB dataset (https://github.com/tangsanli5201/DeepPCB), associated with the paper 'On-line PCB Defect Detector On A New PCB Defect Dataset', using the Kaggle repackaging 'pcb-defect-detection-dataset' by liuxiaolong1. The DeepPCB repository states that the dataset may only be used for research purposes; this project is a non-commercial portfolio demonstration. The current version of this repository contains no dataset images.

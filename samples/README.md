# Sample images

One demo image per defect class, served by `GET /samples` and `GET /samples/{name}`.
Chosen by `scripts/pick_samples.py` from the label files: the image with the fewest
other defect types labelled in it.

**Source:** DeepPCB dataset, validation split (`data/DeepPCB/valid`) of the Kaggle repackaging `liuxiaolong1/pcb-defect-detection-dataset` that this model was evaluated on. Original dataset: https://github.com/tangsanli5201/DeepPCB

**Licence:** DeepPCB dataset, taken from the Kaggle repackaging pcb-defect-detection-dataset by liuxiaolong1, listed on Kaggle as CC0: Public Domain. Six validation-split images are included for demonstration only.

| Sample | Class | Original file | Labelled defects in the image |
|---|---|---|---|
| `1_open_01_PCB__1257.jpg` | Open | `01_PCB__1257.jpg` | 1 Open, 2 Mousebite |
| `2_short_01_PCB__1003.jpg` | Short | `01_PCB__1003.jpg` | 1 Open, 3 Short, 1 Spur |
| `3_mousebite_01_PCB__1304.jpg` | Mousebite | `01_PCB__1304.jpg` | 2 Mousebite, 1 Copper, 1 Pin-hole |
| `4_spur_01_PCB__1484.jpg` | Spur | `01_PCB__1484.jpg` | 1 Open, 1 Short, 2 Spur |
| `5_copper_01_PCB__399.jpg` | Copper | `01_PCB__399.jpg` | 1 Open, 1 Short, 1 Copper |
| `6_pin-hole_01_PCB__626.jpg` | Pin-hole | `01_PCB__626.jpg` | 2 Spur, 2 Pin-hole |

# Training

1. `dataset_prep.py` checks the YOLO export of PKU-Market-PCB: clears label caches, fixes misnamed label files, prints class counts.
2. `train.py base` trains YOLOv8s (SiLU) for 100 epochs.
3. `train.py leaky` rebuilds the network with LeakyReLU(26/256) and fine-tunes from the base weights.
4. `eval.py` reports mAP50, mAP50-95, precision and recall on the validation or test split.

The dataset itself is not included. Set its location in `data.yaml`.

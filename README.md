# Pothole Detection CNN

A convolutional neural network, trained from scratch in PyTorch, that classifies
road-surface photographs as containing a **pothole** or **no pothole**.

## Overview

Potholes are one of the most common and costly forms of road damage, and manual
inspection doesn't scale to an entire city's street network. This project trains
a small CNN to flag potholes automatically from street-level photographs, as a
first step toward automating that inspection process.

The harder (and more useful) version of this problem isn't "pothole vs. a perfect
road" — it's telling a pothole apart from other pavement defects like cracks,
which look visually similar (dark, irregular regions on asphalt). This project
deliberately uses that harder framing: the negative class contains real
pavement cracks, not just clean road.

## Dataset

[Cracks and Potholes in Road Images Dataset](https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset)
(Passos et al., 2020) — 2,235 road-surface photographs captured by a Highway
Diagnostic Vehicle for Brazil's National Department of Transport Infrastructure
(DNIT), each shipped with a pixel-level pothole annotation mask. MIT licensed.

`scripts/prepare_dataset.py` derives the binary classification label for each
image directly from its mask (pothole present / not present) and sorts the
images into `data/pothole/` and `data/no_pothole/`:

| class | images |
|---|---|
| pothole | 564 |
| no_pothole (incl. crack-only roads) | 1,671 |

The raw dataset isn't committed to this repo (it's regenerated on demand); run
the script below to fetch it.

```bash
python scripts/prepare_dataset.py
```

## Model

A compact CNN built from scratch (no pretrained backbone):

```
4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool]  (3->32->64->128->128 channels)
Global Average Pooling
Dropout(0.3) -> Linear(128 -> 2)
```

Input images are resized to 128x128 and normalized with ImageNet statistics.
Training uses class-weighted cross-entropy (to counter the 1:3 pothole /
no-pothole imbalance), Adam, and a ReduceLROnPlateau schedule, with early
stopping on validation loss. See [src/model.py](src/model.py) and
[src/train.py](src/train.py).

Data is split 70 / 15 / 15 (train / val / test), stratified by class, with a
fixed random seed for reproducibility.

## Results

Evaluated on a held-out test set of 336 images never seen during training:

| metric | value |
|---|---|
| Accuracy | 0.744 |
| Precision | 0.496 |
| Recall | 0.753 |
| F1 | 0.598 |
| ROC-AUC | 0.835 |

![Training curves](results/training_curves.png)
![Confusion matrix](results/confusion_matrix.png)

Precision is modest at the default threshold — the model, biased by class
weighting toward catching potholes, produces a meaningful number of false
positives on crack-only road segments. The ROC-AUC of 0.84 indicates good
underlying separability between the two classes; a stricter decision threshold
would trade recall for precision depending on the deployment goal (e.g. city
maintenance crews likely prefer higher recall to avoid missing real potholes).

## Project structure

```
CNN STREET/
├── scripts/prepare_dataset.py   # builds data/ from the public source dataset
├── src/
│   ├── dataset.py                # dataset loading, transforms, stratified split
│   ├── model.py                  # PotholeCNN architecture
│   ├── train.py                  # training loop + checkpointing + curves
│   ├── evaluate.py                # test-set metrics + confusion matrix
│   └── predict.py                 # run inference on a single image
├── models/best_model.pt          # trained weights (best val loss)
└── results/                       # training curves, confusion matrix, metrics.json
```

## Usage

```bash
pip install -r requirements.txt

# 1. build the dataset
python scripts/prepare_dataset.py

# 2. train
python src/train.py --epochs 30

# 3. evaluate on the test split
python src/evaluate.py

# 4. classify a single image
python src/predict.py path/to/photo.jpg
```

## Citation

If you use the underlying dataset, please cite:

> Passos, B.T., Cassaniga, M.J., Fernandes, A.M.R., Medeiros, K.B., Comunello, E.
> "Cracks and Potholes in Road Images Dataset." University of Itajaí Valley (UNIVALI), 2020.

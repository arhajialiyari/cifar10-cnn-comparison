# CIFAR-10: Comparison of Three Convolutional Neural Networks

A reproducibility-oriented refactor of three CNN experiments in an earlier
[Google Colab notebook](https://colab.research.google.com/drive/1UwTCG81bCByh1s3R_jvjYYqM9DQRqgvZ).
The original coursework compared a ReLU CNN, a deeper batch-normalized CNN,
and a Tanh CNN. This repository retains their broad architectures and original
optimizer/batch-size choices but fixes the evaluation protocol.

## Verification status

- **Smoke test passed on Windows (Python 3.12.10).** All three CNN configurations ran for one epoch on stratified subsets of 3,000 training and 500 validation images. The script selected a checkpoint by validation loss and wrote the smoke reports to `results/smoke/`.
- **Full training and official test evaluation are pending.** The smoke test checks the training, selection, and reporting workflow. It skips evaluation on the official test set and provides no held-out performance estimate.

## Experimental design

- **Data:** CIFAR-10 (50,000 training images and 10,000 official test images; 10 classes).
  The loader reads pre-existing local CIFAR-10 batch files without downloading. No dataset is committed.
- **Split:** Stratified, seeded 45,000/5,000 training/validation partition from
  the official 50,000 training images; the official test set is kept untouched
  for model selection.
- **Inputs:** Images scaled to `[0, 1]`. Sparse integer labels avoid unnecessary
  one-hot conversion.
- **Models:** (1) five-layer ReLU CNN with dense head and RMSprop;
  (2) eight-layer ReLU CNN with batch normalization and Adamax;
  (3) seven-layer Tanh CNN with dense head and Adamax.
- **Selection:** Lowest **validation cross-entropy** among the best checkpoints
  for the three models. Early stopping restores the best validation-loss epoch.
- **Final evaluation:** Evaluate the selected model on the official CIFAR-10
  test set **once**, reporting loss, accuracy, macro precision/recall/F1 and a
  confusion matrix. No test metrics are used in model selection.

These are **different model-and-training recipes**, not a controlled ablation
of batch normalization or activation function: depth, optimizer, learning rate,
and batch size also differ. Fixed random seeds improve repeatability but do not
guarantee exact determinism across machines, GPUs, or TensorFlow versions.

## Repository files

- `cifar10_experiment.py` — training, validation-based selection, final test
  evaluation, plots and JSON reports.
- `cifar10_cnn_comparison.ipynb` — lightweight notebook runner and interpretation.
- `requirements.txt` — tested-for-syntax dependency specification for Python 3.12.
- `results/` — ignored local output directory created by running the experiment.

## Setup (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a **smoke test** (small stratified subsets, one epoch per model). This
checks mechanics only and **does not evaluate the official test set**:

```powershell
.\.venv\Scripts\python.exe cifar10_experiment.py --smoke
```

Run the full original 50/100/150-epoch budgets (can require substantial compute):

```powershell
.\.venv\Scripts\python.exe cifar10_experiment.py --epochs 50 100 150
```

For full runs, `results/full/` contains per-model histories, learning curves,
checkpoint files, validation comparison, `selected_model.json`, and (only after
successful model selection) `test_metrics.json` and `confusion_matrix.png`.
The dataset and model outputs are not committed.

## Important limitation of the historical notebook

The original Colab version used the **official test set as validation data in
all three model fits**, so its reported `val_accuracy` values are **not
unbiased held-out test estimates**. The original model histories cannot be
compared directly with the corrected results from this project. No such old
scores have been copied into the new experiment's results.

## Provenance

Refactored from the author's deep learning course exercises. The linked original
notebook is the source of the three architectures; the data-splitting,
checkpointing, selection, reporting and reproducibility workflow are new.

## Offline dataset requirement

The experiment requires the extracted CIFAR-10 Python-format dataset
inside the standard Keras dataset cache:

`~/.keras/datasets/cifar-10-batches-py/`

The directory must contain `data_batch_1` through `data_batch_5`
and `test_batch`. `KERAS_HOME` can override the Keras cache location.

The experiment does not download missing data automatically.

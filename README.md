# AF-Mamba: Efficient Long-Term Signal Modeling for Early Prediction of Atrial Fibrillation Onset

## Overview

AF-Mamba is a deep learning framework for predicting atrial fibrillation (AF) onset **1 hour in advance** from long-term RR interval (RRI) sequences.

The model combines:

- **Temporal Convolutional Networks (TCNs)** for local temporal feature extraction
- **Mamba selective state-space modeling** for efficient long-range sequence modeling
- **Global average and max pooling** for pre-AF classification

The primary model uses a **1-hour RRI input window from 2 h to 1 h before AF onset**, corresponding to a **1-hour prediction horizon**.

In subject-wise 5-fold testing, AF-Mamba achieved:

| Metric | Performance |
|---|---:|
| Sensitivity | 0.889 |
| Specificity | 0.943 |
| F1-score | 0.813 |
| AUROC | 0.974 |
| AUPRC | 0.933 |

Paired cross-dataset holdout evaluation across unseen AF and NSR source datasets achieved a mean AUROC of **0.897**.

---

## Repository Structure

```text
AF_Mamba/
├── Data/
│   ├── structured_dataset_1hz.pt
│   ├── structured_dataset_4hz.pt
│   └── subject_folds.pt
├── Models/
│   ├── AF_mamba.py
│   ├── baselines.py
│   ├── other_models.py
│   └── third_party/
├── Results/
│   ├── oof_ectopy_predictions.csv
│   └── gee_oof_mamba_hrv.csv
├── Trained_Models/
│   ├── af_mamba/
│   └── ...
├── train.ipynb
├── test.ipynb
├── test_ablation.ipynb
├── test_hrv.ipynb
├── test_paired_holdout.ipynb
├── test_operating_points.ipynb
├── analysis_ectopy_gee.ipynb
├── data_utils.py
├── evaluation.py
├── hrv_utils.py
└── requirements.txt
```

### Main notebooks

- `train.ipynb` — trains AF-Mamba using subject-wise 5-fold cross-validation.
- `test.ipynb` — evaluates pretrained models and includes window-level and subject-level evaluation.
- `test_ablation.ipynb` — evaluates AF-Mamba and alternative architectures across 5-, 10-, 30-, and 60-minute RRI inputs.
- `test_hrv.ipynb` — evaluates the HRV-XGBoost baseline using 4-Hz RRI features.
- `test_paired_holdout.ipynb` — evaluates paired cross-dataset generalization with one AF and one NSR dataset held out together.
- `test_operating_points.ipynb` — evaluates calibration, Brier score, reliability curve, PPV/NPV, and sensitivity–false-alert trade-offs.
- `analysis_ectopy_gee.ipynb` — evaluates AF-Mamba across ectopic-burden groups and performs HRV-adjusted generalized estimating equation (GEE) analyses.

---

## Dataset

The study uses five public long-term Holter/ambulatory RRI datasets:

- IRIDIA-AF
- Long-Term AF Database (LTAF)
- MIT-BIH Atrial Fibrillation Database
- MIT-BIH Normal Sinus Rhythm Database
- Normal Sinus Rhythm RR Interval Database

The final cohort contains **232 subjects**:

- 160 AF subjects
- 72 NSR subjects

Using strict non-overlapping 1-hour windows, the primary analysis contains:

- 256 pre-AF windows
- 1,326 NSR windows
- 1,582 windows in total

Please obtain the original datasets from their respective providers and follow the corresponding data-use and redistribution terms.

---

## Preprocessing

For the primary 1-Hz model input:

1. Recordings are split at discontinuities with RR intervals greater than 10 s.
2. RR intervals outside 0.2–5 s are removed.
3. Remaining RRIs are interpolated at 1 Hz using cubic spline interpolation.
4. Pre-AF inputs are extracted from **2 h to 1 h before AF onset**.
5. NSR recordings are divided into strict non-overlapping 1-hour windows.

A corresponding **4-Hz RRI representation** is used for HRV-based analyses.

All subjects are assigned to mutually exclusive subject-wise folds, and no subject appears in more than one of the training, validation, or test splits within a fold.

---

## Model

AF-Mamba consists of:

- 3 residual TCN blocks
- 32 channels
- dilation rates of 1, 2, and 4
- kernel size of 3
- dropout of 0.2
- Mamba block with `d_model=32`
- convolutional feed-forward network
- global average pooling + global max pooling
- fully connected classification head

The selected training configuration uses:

```text
Batch size:       16
Learning rate:    1e-4
Weight decay:     1e-4
Optimizer:        AdamW
Early stopping:   patience = 10
Maximum epochs:   1000
Sampling rate:    1 Hz
```

---

## Installation

```bash
git clone https://github.com/yongbin98/AF_Mamba.git
cd AF_Mamba
pip install -r requirements.txt
```

Reference environment:

```text
Python:  3.12.12
PyTorch: 2.4.0+cu121
CUDA:    12.1
```

GPU support is recommended for training and deep-learning inference.

---

## Evaluation

To reproduce the primary AF-Mamba evaluation, use the pretrained checkpoints in:

```text
Trained_Models/af_mamba/
```

and run:

```text
test.ipynb
```

The decision threshold is selected using the validation set and then applied unchanged to the corresponding test set.

Additional analyses can be reproduced using the corresponding notebooks listed above.

---

## Reproducibility

Random seeds are fixed where applicable. However, retraining results may vary slightly across hardware and software environments because some GPU/CUDA operations can be nondeterministic.

For direct reproduction of the reported evaluation results, using the provided pretrained model checkpoints is recommended.

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{lee2026afmamba,
  title={AF-Mamba: Efficient Long-Term Signal Modeling for Early Prediction of Atrial Fibrillation Onset},
  author={Lee, Yongbin and Chon, Ki H.},
  year={2026},
  note={Manuscript under review}
}
```

Citation information will be updated upon publication.

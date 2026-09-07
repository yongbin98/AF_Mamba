# AF-Mamba: Early Prediction of Atrial Fibrillation Onset

## 🧠 Overview

AF-Mamba is a deep learning framework for predicting atrial fibrillation (AF) onset from long-term RR interval (RRI) sequences.

The primary task uses a **1-hour RRI segment from 2 h to 1 h before AF onset** to predict AF **1 hour in advance**.

AF-Mamba combines:

- Temporal Convolutional Networks (TCNs) for local temporal feature extraction
- Mamba selective state-space modeling for long-range sequence modeling
- Global average/max pooling for pre-AF classification

---

## 📁 Repository Structure

```text
Data/                        # Processed RRI datasets and subject folds
Models/                      # AF-Mamba and baseline architectures
Results/                     # Saved OOF analysis inputs
Trained_Models/              # Pretrained model checkpoints

train.ipynb                  # AF-Mamba training
test.ipynb                   # Main model evaluation
test_ablation.ipynb          # Input-length ablation
test_hrv.ipynb               # HRV-XGBoost baseline
test_paired_holdout.ipynb    # Paired cross-dataset holdout
test_operating_points.ipynb  # Calibration and operating-point analysis
analysis_ectopy_gee.ipynb    # Ectopy burden and HRV-adjusted GEE

data_utils.py                # Data loading and split utilities
evaluation.py                # Evaluation utilities
hrv_utils.py                 # HRV feature utilities
requirements.txt             # Python dependencies
```

---

## 📊 Datasets

This study uses five publicly available long-term Holter/ambulatory datasets:

- **IRIDIA-AF**  
  https://zenodo.org/records/8405941

- **Long-Term Atrial Fibrillation Database (LTAF)**  
  https://physionet.org/content/ltafdb/1.0.0/

- **MIT-BIH Atrial Fibrillation Database**  
  https://physionet.org/content/afdb/1.0.0/

- **MIT-BIH Normal Sinus Rhythm Database**  
  https://physionet.org/content/nsrdb/1.0.0/

- **Normal Sinus Rhythm RR Interval Database**  
  https://physionet.org/content/nsr2db/1.0.0/

Raw datasets are not redistributed here. Please obtain them from the original sources and follow their data-use terms.

---

## ⚙️ Google Colab Setup

A reference environment used for this project was based on a **late-2025 Google Colab runtime** with:

```text
Python 3.12.12
PyTorch 2.4.0 + CUDA 12.1
```

For a fresh Colab runtime, run the following installation cell before importing PyTorch/Mamba:

```python
!pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
!pip uninstall -y -q mamba-ssm causal-conv1d
!pip install -q causal-conv1d==1.4.0
!pip install -q mamba-ssm==2.2.2
!pip install -q torchinfo fvcore
!pip install -q fastcore==1.14.5
!pip install -q fastai==2.8.4
!pip install -q tsai==0.4.1 --no-deps
!pip install -q einops
```

After installation, **restart the Colab runtime once**, then run the notebooks.

Alternatively:

```bash
pip install -r requirements.txt
```

Exact package versions are listed in `requirements.txt`.

---

## 🚀 Usage

### Train AF-Mamba

```text
train.ipynb
```

### Main evaluation

```text
test.ipynb
```

### Additional analyses

```text
test_ablation.ipynb
test_hrv.ipynb
test_paired_holdout.ipynb
test_operating_points.ipynb
analysis_ectopy_gee.ipynb
```

Pretrained checkpoints are stored under:

```text
Trained_Models/
```

---

## 📝 Notes

- Subject-wise folds are shared across the 1-Hz and 4-Hz RRI datasets.
- The main model uses 1-Hz RRI signals.
- HRV-based analyses use the corresponding 4-Hz RRI representation.
- Random seeds are fixed where applicable, but retraining may vary slightly across GPU/CUDA environments.
- For direct reproduction of reported evaluation results, use the provided pretrained checkpoints.

---

## 📦 Processed Data

The processed RRI datasets and large pretrained model files are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22572206.svg)](https://doi.org/10.5281/zenodo.22572206)

**Zenodo:** https://doi.org/10.5281/zenodo.22572206

Download the files and place them in the corresponding directories:

```text
Data/
├── structured_dataset_1hz.pt
├── structured_dataset_4hz.pt
├── structured_dataset_ectopic.pt
└── subject_folds.pt

Trained_Models/
└── resnet1d/

---

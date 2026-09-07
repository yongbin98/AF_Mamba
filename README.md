# AF-Mamba: Early Prediction of Atrial Fibrillation Onset

## Overview

AF-Mamba is a deep learning framework for **early prediction of atrial fibrillation (AF) onset** from long RR interval (RRI) sequences.

This repository includes:

- AF-Mamba model implementation
- Baseline models
- Training and testing notebooks
- Ablation, HRV baseline, paired holdout, operating point, ectopy, and GEE analyses

The primary task is to predict AF onset using a **1-hour RRI segment from 2 h to 1 h before AF onset**.

---

## 📁 Repository Structure

```text
Data/                        # Processed datasets and subject fold file
Models/                      # AF-Mamba, baselines, and other model definitions
Results/                     # Optional saved analysis outputs
Trained_Models/              # Pretrained model checkpoints

train.ipynb                  # Training notebook
test.ipynb                   # Main test notebook
test_ablation.ipynb          # Input-length ablation testing
test_hrv.ipynb               # HRV-XGBoost baseline testing
test_paired_holdout.ipynb    # Paired cross-dataset holdout testing
test_operating_points.ipynb  # Calibration, reliability, and operating-point analysis
analysis_ectopy_gee.ipynb    # Ectopy burden and HRV-adjusted GEE analysis

data_utils.py                # Dataset loading and split utilities
evaluation.py                # Evaluation utilities
hrv_utils.py                 # HRV feature extraction utilities
requirements.txt             # Dependencies

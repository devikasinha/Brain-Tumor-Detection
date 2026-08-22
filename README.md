Brain Tumor Detection using Deep Learning & Computer Vision

This project focuses on the automated detection and classification of brain tumors from MRI scans to aid medical diagnostics. By leveraging deep learning models alongside computer vision techniques, the pipeline processes complex medical imagery to accurately identify tumor regions while minimizing false positives.

Image Preprocessing & Noise Removal: Applies advanced image filtering, contrast adjustments, and spatial smoothing to clean raw MRI scans and enhance feature clarity before training.

Deep Learning Architecture: Employs Convolutional Neural Networks (CNNs) for precise feature extraction and classification of MRI slices.

High Performance: Achieves a 89% classification accuracy, demonstrating robust generalization across diverse scan samples.

Diagnostic Support: Designed to assist healthcare professionals by streamlining the initial screening phase and speeding up image assessment.# Multi-Model Hyperparameter Tuning Pipeline

A clean, modular Python implementation for hyperparameter optimization across ensemble and gradient boosted decision tree architectures using **5-Fold Cross-Validation** via Scikit-Learn's `GridSearchCV`.

## Overview

This repository demonstrates how to build an end-to-end training and evaluation pipeline comparing three popular machine learning models:
* **Random Forest** (Scikit-Learn)
* **XGBoost Classifier** (`xgboost`)
* **LightGBM Classifier** (`lightgbm`)

The script generates synthetic classification data, runs grid search cross-validation across hyperparameter space, selects optimal model parameters, and evaluates generalization performance on holdout test data.

## Features

* **5-Fold Cross-Validation:** Robust performance estimation preventing data leakage.
* **Automated Grid Search:** Systematic hyperparameter selection for depth, tree count, and learning rates.
* **Parallel Processing:** Configured with `n_jobs=-1` for multi-core execution.
* **Performance Comparison:** Generates side-by-side metric tables for CV and Test set results.





Running GridSearch for Random Forest...
Best Parameters for Random Forest: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
Best 5-Fold CV Score: 0.8875

Running GridSearch for XGBoost...
Best Parameters for XGBoost: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
Best 5-Fold CV Score: 0.8925

Running GridSearch for LightGBM...
Best Parameters for LightGBM: {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100, 'num_leaves': 31}
Best 5-Fold CV Score: 0.8950

           Model  CV Score  Test Accuracy
0  Random Forest    0.8875           0.890
1        XGBoost    0.8925           0.895
2       LightGBM    0.8950           0.900

```bash
pip install numpy pandas scikit-learn xgboost lightgbm

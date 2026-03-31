# IMDb Advanced Analytics, Time Series & XAI

## Overview
This repository contains an advanced Data Mining and Machine Learning project based on an expanded **IMDb database** (over 150,000 observations across 45 variables) and a dedicated time series dataset. The workflow tackles complex real-world data challenges, including severe class imbalance, multidimensional outlier detection, Time Series classification, and Explainable AI (XAI). **[Read the full Project Report (PDF)](IMDbAdvancedProjectReport.pdf)**

## Project Workflow & Architecture

### 1. Advanced Data Pre-processing & Outlier Detection
* **Pipeline Refinement**: Engineered geographic data, handled complex missing values via Gaussian Mixture Models (GMM) and removed low-information noise.
* **Ensemble Outlier Detection**: Implemented a robust, multi-algorithm strategy to isolate anomalies using Local Outlier Factor (LOF), Angle-Based Outlier Detection (ABOD), and Isolation Forest (IF). Isolated 1.35% structural anomalies, validated via PCA and t-SNE projections.

### 2. Imbalanced Learning & Classification
* Tackled severe class imbalances (e.g., `tvSeries` vs. `Video`, top vs. low ratings) using specialized sampling techniques.
* Trained and evaluated high-performance classifiers including **Logistic Regression, SVM, Ensembles (Bagging, Random Forest, AdaBoost), XGBoost and Neural Networks**.
* Reached F1-Weighted: 0.92, ROC-AUC: 0.988 on titleType classification using optimized XGBoost.

### 3. Explainable AI (XAI)
* Integrated **SHAP** and **LORE** (66.2% rule accuracy) to interpret the predictive logic of "black-box" models.
* Extracted human-readable explanations to understand feature impact on IMDb ratings and title classifications.

### 4. Time Series Analysis
* Clustering: Segmented box office revenue behaviors using Hierarchical Clustering combined with Sakoe-Chiba constrained DTW.
* Motifs & Shapelets: Extracted >4,300 temporal motifs via Matrix Profile (5-day window) and matched them against algorithmic Shapelets to find discriminative sequences.
* Temporal Classification: Evaluated Shapelet-based Random Forest, ROCKET (identified severe overfitting), and k-NN. The most robust generalization was achieved by k-NN with DTW + Sakoe-Chiba constraint (Macro F1: 0.46, lowest train-test gap: 0.045).

## Tech Stack
* **Language**: Python (Pandas, Matplotlib, Seaborn, NumPy, SciPy)
* **Advanced Machine Learning**: Scikit-Learn, XGBoost, TensorFlow, Keras
* **Time Series Analysis**: Tslearn, Sktime
* **Explainable AI (XAI)**: SHAP, LORE

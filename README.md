# IMDb Advanced Analytics, Time Series & XAI

## Overview
This repository contains an advanced Data Mining and Machine Learning project based on an expanded **IMDb database** (over 130,000 observations across 45 variables). The workflow tackles complex real-world data challenges, including severe class imbalance, multidimensional outlier detection, Time Series classification, and Explainable AI (XAI). **[Read the full Project Report (PDF)](IMDbAdvancedProjectReport.pdf)**

## Project Workflow & Architecture

### 1. Advanced Data Pre-processing & Outlier Detection
* **Pipeline Refinement**: Engineered geographic data, handled complex missing values, and removed low-information noise.
* **Ensemble Outlier Detection**: Implemented a robust, multi-algorithm strategy to isolate anomalies using Local Outlier Factor (LOF), Angle-Based Outlier Detection (ABOD), and Isolation Forest (IF).

### 2. Imbalanced Learning & Classification
* Tackled severe class imbalances (e.g., `tvSeries` vs. `Video`, top vs. low ratings) using specialized sampling techniques.
* Trained and evaluated high-performance classifiers including **Logistic Regression, SVM, Ensembles (Bagging), XGBoost, and Neural Networks**.

### 3. Explainable AI (XAI)
* Integrated **SHAP** and **LORE** to interpret the predictive logic of "black-box" models.
* Extracted human-readable explanations to understand feature impact on IMDb ratings and title classifications.

### 4. Time Series Analysis
Processed and analyzed sequential/temporal data associated with IMDb records:
* **Time Series Clustering**: Utilized sequence data for temporal outlier detection.
* **Motifs & Shapelets**: Extracted Time Series motifs and shapelets to identify recurring temporal patterns.
* **Advanced Temporal Classification**: Deployed Shapelets-based models, K-NN, and the **ROCKET** (RandOm Convolutional KErnel Transform) algorithm for state-of-the-art time series classification.

## Tech Stack
* **Language**: Python
* **Advanced Machine Learning**: Scikit-Learn, XGBoost, Neural Networks
* **Time Series Analysis**: ROCKET, Shapelets extraction
* **Explainable AI (XAI)**: SHAP, LORE

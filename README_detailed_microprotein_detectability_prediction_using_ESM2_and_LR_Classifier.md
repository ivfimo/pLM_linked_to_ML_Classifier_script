# README: Microprotein Detectability Prediction Using ESM2 and Logistic Regression

## Overview
This script implements a machine learning pipeline that integrates the ESM2 transformer-based protein model with a Logistic Regression classifier to predict the detectability of microprotein sequences. The script preprocesses data, extracts embeddings from sequences, performs feature selection, trains a logistic regression model, and evaluates its performance using multiple metrics.

## Requirements
### Dependencies
Ensure the following Python libraries are installed:
```bash
pip install numpy pandas torch transformers tqdm scikit-learn matplotlib seaborn shap
```
### Hardware Requirements
- A CUDA-compatible GPU is recommended for efficient embedding generation.
- Sufficient RAM for dataset processing.

## Step-by-Step Workflow

### 1. Load and Preprocess Dataset
- The dataset is loaded from a CSV file (`seq7264ORFs_dataset.csv`).
- Missing values are removed.
- Numerical features are extracted (excluding `sequence` and `group`).
- A balanced dataset is created by undersampling negative samples to match the number of positive samples.

### 2. Data Splitting
- The dataset is split into training, validation, and test sets (80-20% split, then further 15% validation split from training data).
- Sequences and numerical features are stored separately for further processing.

### 3. Load Pre-trained ESM2 Model
- The `facebook/esm2_t6_8M_UR50D` model is loaded from the Hugging Face transformers library.
- The model is set to evaluation mode and transferred to GPU (if available).

### 4. Generate Sequence Embeddings
- A function `generate_embeddings` tokenizes and processes sequences in batches.
- The mean of the last hidden state from ESM2 is extracted as the embedding.
- Embeddings for training, validation, and test sets are saved as `.npy` files.

### 5. Combine Features
- The numerical features are combined with the sequence embeddings to create the final feature set.

### 6. Hyperparameter Tuning for Logistic Regression
- Grid search with cross-validation is used to tune `C` and `solver` hyperparameters.
- The best model is selected based on the highest AUC-ROC score.

### 7. Model Evaluation
- The selected Logistic Regression model is evaluated on the test set.
- Predictions are generated, and classification reports are printed.
- ROC and Precision-Recall curves are plotted.

### 8. Explainability Analysis
- SHAP (SHapley Additive Explanations) values are computed to interpret model predictions.
- Permutation importance is used to rank feature contributions.

### 9. Visualization
- The following visualizations are generated:
  - SHAP summary plot.
  - Permutation importance plot (Top 50 features).
  - ROC curve.
  - Precision-Recall curve.
  - Confusion matrix heatmap.

## Outputs
- **Best model parameters** (printed after GridSearchCV tuning)
- **SHAP values summary plot**
- **Permutation importance rankings**
- **Evaluation metrics** (precision, recall, AUC-ROC, confusion matrix)

## Usage
To run the script:
```bash
python your_script.py
```
Ensure the dataset (`seq7264ORFs_dataset.csv`) is placed in the correct path.


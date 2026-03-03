
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import shap
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("/content/seq7264ORFs_dataset.csv")

# Drop missing values
data = data.dropna()

# Select all numerical features excluding 'sequence' and 'group'
numerical_features = [col for col in data.columns if col not in ['sequence', 'group']]
X_features = data[numerical_features]

# Balance dataset
positive_samples = data[data['group'] == 1].sample(n=1785, random_state=42)
negative_samples = data[data['group'] == 0].sample(n=1785, random_state=42)
balanced_data = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42)

# Extract sequences and features
X_seq = balanced_data['sequence']
X_features = balanced_data[numerical_features]
y = balanced_data['group'].values

# Split dataset
X_train_seq, X_test_seq, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
    X_seq, X_features, y, test_size=0.2, stratify=y, random_state=42
)
X_train_seq, X_val_seq, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
    X_train_seq, X_train_feat, y_train, test_size=0.15, stratify=y_train, random_state=42
)
# Load pre-trained protein model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
model.eval()

def generate_embeddings(sequences, batch_size=1, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(outputs)
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    return np.vstack(embeddings).astype(np.float32)

# Generate and save embeddings
for split_name, X_split in zip(["train", "val", "test"], [X_train_seq, X_val_seq, X_test_seq]):
    file_path = f"/content/{split_name}_embeddings.npy"
    if not os.path.exists(file_path):
        np.save(file_path, generate_embeddings(X_split.tolist()))

def load_embeddings(path):
    return np.load(path, mmap_mode='r')

X_train_emb = load_embeddings("/content/train_embeddings.npy")
X_val_emb = load_embeddings("/content/val_embeddings.npy")
X_test_emb = load_embeddings("/content/test_embeddings.npy")

# Combine embeddings with additional features
X_train = np.hstack((X_train_emb, X_train_feat.values))
X_val = np.hstack((X_val_emb, X_val_feat.values))
X_test = np.hstack((X_test_emb, X_test_feat.values))

# Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}
clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
clf.fit(X_train, y_train)

# Best model
best_model = clf.best_estimator_
print("Best hyperparameters:", clf.best_params_)

# Evaluate model
y_pred_probs = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_probs >= 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=["Non-Detectable", "Detectable"]))


# ROC Curve
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = roc_auc_score(y_test, y_pred_probs)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# PR Curve
plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Detectable", "Detectable"],
            yticklabels=["Non-Detectable", "Detectable"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# SHAP Analysis

print("\n" + "="*60)
print("EXTRACTING ORIGINAL FEATURES AND SHAP ANALYSIS WITH ORIGINAL NAMES")
print("="*60)

# Load the original dataset to get feature names
original_data_path = "/content/Chimp_Table_S8_ List_of_7264ncORFs_along_with_ORBLv_45features.csv"
original_data = pd.read_csv(original_data_path)

# Get original feature names (excluding sequence and group columns)
original_feature_names = [col for col in original_data.columns if col not in ['sequence', 'group']]
print(f"\nFound {len(original_feature_names)} original features:")
print(original_feature_names[:10])  # Print first 10 as sample

# Create a mapping of original feature names to their indices in the combined feature set
# Note: The combined features are [embeddings + original_features]
n_embeddings = X_train_emb.shape[1]
original_feature_indices = list(range(n_embeddings, n_embeddings + len(original_feature_names)))

# Create complete feature names list
all_feature_names = [f"ESM2_Embedding_{i}" for i in range(n_embeddings)] + original_feature_names

# SHAP analysis using the best logistic regression model
print("\n" + "="*60)
print("PERFORMING SHAP ANALYSIS ON LOGISTIC REGRESSION MODEL")
print("="*60)

# Create SHAP explainer for logistic regression
explainer_lr = shap.LinearExplainer(best_model, X_train, feature_perturbation="correlation_dependent")

# Calculate SHAP values for test set
print("Calculating SHAP values for test set...")
shap_values_lr = explainer_lr.shap_values(X_test)

# Create SHAP summary plot with ALL features (including original names)
plt.figure(figsize=(16, 10))
shap.summary_plot(
    shap_values_lr,
    X_test,
    feature_names=all_feature_names,
    show=False,
    max_display=30,
    plot_size=(16, 10)
)
plt.title("SHAP Feature Importance - Logistic Regression (All Features)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("/content/shap_summary_all_features_lr.png", dpi=300, bbox_inches='tight')
plt.show()
print("SHAP summary plot saved to /content/shap_summary_all_features_lr.png")

# Create SHAP summary plot with ONLY original features
print("\n" + "="*60)
print("CREATING SHAP SUMMARY PLOT WITH ONLY ORIGINAL FEATURES")
print("="*60)

# Extract SHAP values for original features only
shap_values_original = shap_values_lr[:, original_feature_indices]
X_test_original = X_test[:, original_feature_indices]

# Create SHAP summary plot for original features only
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_values_original,
    X_test_original,
    feature_names=original_feature_names,
    show=False,
    max_display=30,
    plot_size=(14, 10),
    color=plt.cm.RdBu
)
plt.title("SHAP Feature Importance - Original Biological Features Only", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("/content/shap_summary_original_features_only.png", dpi=300, bbox_inches='tight')
plt.show()
print("Original features SHAP summary plot saved to /content/shap_summary_original_features_only.png")

# Create a bar plot of SHAP importance for original features
print("\n" + "="*60)
print("CREATING SHAP BAR PLOT FOR ORIGINAL FEATURES")
print("="*60)

# Calculate mean absolute SHAP values for original features
mean_abs_shap = np.abs(shap_values_original).mean(axis=0)

# Create dataframe for original feature importance
original_importance_df = pd.DataFrame({
    'Feature': original_feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

# Save to CSV
original_importance_df.to_csv('/content/original_features_shap_importance.csv', index=False)
print(f"Original feature importance saved to /content/original_features_shap_importance.csv")

# Plot top 30 original features by SHAP importance
plt.figure(figsize=(14, 10))
top_n = min(30, len(original_importance_df))
top_features = original_importance_df.head(top_n)

# Create horizontal bar plot
colors = plt.cm.viridis(np.linspace(0, 1, top_n))
plt.barh(range(top_n), top_features['Mean_Abs_SHAP'].values, color=colors[::-1])
plt.yticks(range(top_n), top_features['Feature'].values)
plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12)
plt.ylabel('Original Features', fontsize=12)
plt.title(f'Top {top_n} Original Features by SHAP Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at top
plt.tight_layout()
plt.savefig("/content/original_features_shap_importance_bar.png", dpi=300, bbox_inches='tight')
plt.show()


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


# SHAP Analysis
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)


# Permutation Importance
perm_importance = permutation_importance(best_model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({'Feature': X_features.columns.tolist() + [f"emb_{i}" for i in range(X_train_emb.shape[1])], 'Importance': perm_importance.importances_mean})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(50)

plt.figure(figsize=(40, 20))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette="viridis")
plt.xlabel("Mean Importance Score")
plt.ylabel("Feature")
plt.title("Top 50 Permutation Importance Features")
plt.show()

# Visualization
plt.figure(figsize=(60, 20))

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
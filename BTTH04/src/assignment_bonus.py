import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.svm_sgd import LinearSoftMarginSVM

df = pd.read_csv("../BTTH01/data_bds.csv")
df = df.drop_duplicates().reset_index(drop=True)

a_all = np.where(df["price"].values >= 500000000, 1, -1)

keywords = [
    "bán", "cho thuê", "sổ hồng", "tỷ", "tháng",
    "phòng trọ", "cọc", "thổ cư", "sang tên", "giá thuê",
    "sinh viên", "giờ giấc", "chính chủ", "hẻm", "mặt tiền"
]

text_series = (df["title"].fillna("") + " " + df["body"].fillna("")).str.lower()

kw_rows = []
for txt in text_series:
    row_feats = []
    for kw in keywords:
        if kw in txt:
            row_feats.append(1.0)
        else:
            row_feats.append(0.0)
    kw_rows.append(row_feats)

X_kw = np.array(kw_rows, dtype=np.float64)
area_col = df[["area"]].values.astype(np.float64)
cat_dummies = pd.get_dummies(df["category"], dtype=float).values

X_all = np.hstack([area_col, cat_dummies, X_kw])

np.random.seed(42)
idx = np.arange(len(X_all))
np.random.shuffle(idx)

split_point = int(0.8 * len(X_all))
train_idx = idx[:split_point]
test_idx = idx[split_point:]

X_train = X_all[train_idx].copy()
a_train = a_all[train_idx]
X_test = X_all[test_idx].copy()
a_test = a_all[test_idx]

mean_area = np.mean(X_train[:, 0])
std_area = np.std(X_train[:, 0])
X_train[:, 0] = (X_train[:, 0] - mean_area) / std_area
X_test[:, 0] = (X_test[:, 0] - mean_area) / std_area

print("Dataset BDS shape:", X_all.shape, a_all.shape)
print("Train shape:", X_train.shape, a_train.shape)
print("Test shape:", X_test.shape, a_test.shape)

bonus_svm = LinearSoftMarginSVM(
    penalty_c=10.0,
    learning_rate=1e-3,
    num_epochs=150
)

bonus_svm.train(X_train, a_train)
manual_pred = bonus_svm.predict(X_test)
manual_scores = bonus_svm.compute_metrics(a_test, manual_pred)

lib_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
lib_svm.fit(X_train, a_train)
lib_pred = lib_svm.predict(X_test)

lib_scores = {
    "precision": precision_score(a_test, lib_pred, pos_label=1, zero_division=0),
    "recall": recall_score(a_test, lib_pred, pos_label=1, zero_division=0),
    "f1": f1_score(a_test, lib_pred, pos_label=1, zero_division=0)
}

print("=" * 10 + " BONUS DATASET (CHOTOT BDS) - MANUAL SVM " + "=" * 10)
for key in manual_scores:
    print(f"{key}:\t{manual_scores[key]:.4f}")

print("\n" + "=" * 10 + " BONUS DATASET (CHOTOT BDS) - SKLEARN LinearSVC " + "=" * 10)
for key in lib_scores:
    print(f"{key}:\t{lib_scores[key]:.4f}")

os.makedirs("Output_Evidence", exist_ok=True)

cm_bonus = confusion_matrix(a_test, manual_pred, labels=[-1, 1])
plt.figure(figsize=(5, 5))
sns.heatmap(cm_bonus, annot=True, fmt="d", cmap="Oranges", xticklabels=["CHO THUE", "BAN"], yticklabels=["CHO THUE", "BAN"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Bonus Chotot BDS (Manual SVM)")
plt.savefig("Output_Evidence/bonus_confusion_matrix.png", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(bonus_svm.loss_history)
plt.title("Hinge Loss Curve - Bonus Chotot BDS (SGD)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("Output_Evidence/bonus_loss_curve.png", bbox_inches="tight")
plt.show()
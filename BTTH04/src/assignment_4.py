import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.data_preprocessing import prepare_chest_xray_data
from src.svm_sgd import LinearSoftMarginSVM

X_train, a_train = prepare_chest_xray_data(
    data_root="data/chest_xray/train",
    num_per_label=200,
    target_size=512,
    random_seed=42
)

X_test, a_test = prepare_chest_xray_data(
    data_root="data/chest_xray/test",
    num_per_label=100,
    target_size=512,
    random_seed=42
)

custom_svm = LinearSoftMarginSVM(penalty_c=1.0, learning_rate=1e-6, num_epochs=30)
custom_svm.train(X_train, a_train)
custom_pred = custom_svm.predict(X_test)
custom_scores = custom_svm.compute_metrics(a_test, custom_pred)

lib_svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
lib_svm.fit(X_train, a_train)
lib_pred = lib_svm.predict(X_test)

lib_scores = {
    "precision": precision_score(a_test, lib_pred, pos_label=1, zero_division=0),
    "recall": recall_score(a_test, lib_pred, pos_label=1, zero_division=0),
    "f1": f1_score(a_test, lib_pred, pos_label=1, zero_division=0)
}

print("=" * 10 + " MANUAL SVM (SGD - NUMPY) " + "=" * 10)
for key in custom_scores:
    print(f"{key}:\t{custom_scores[key]:.4f}")

print("\n" + "=" * 10 + " SKLEARN SVM (LinearSVC) " + "=" * 10)
for key in lib_scores:
    print(f"{key}:\t{lib_scores[key]:.4f}")

os.makedirs("Output_Evidence", exist_ok=True)

cm_lib = confusion_matrix(a_test, lib_pred, labels=[-1, 1])
plt.figure(figsize=(5, 5))
sns.heatmap(cm_lib, annot=True, fmt="d", cmap="Greens", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Sklearn LinearSVC")
plt.savefig("Output_Evidence/ass4_confusion_matrix.png", bbox_inches="tight")
plt.show()
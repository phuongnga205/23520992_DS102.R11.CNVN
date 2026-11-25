import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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

print("Train shape:", X_train.shape, a_train.shape)
print("Test shape:", X_test.shape, a_test.shape)

svm_model = LinearSoftMarginSVM(
    penalty_c=1.0,
    learning_rate=1e-6,
    num_epochs=30
)

svm_model.train(X_train, a_train)
a_pred = svm_model.predict(X_test)
eval_scores = svm_model.compute_metrics(a_test, a_pred)

print("=" * 10 + " ASSIGNMENT 3: SOFT-MARGIN SVM (SGD - NUMPY) " + "=" * 10)
for key in eval_scores:
    print(f"{key}:\t{eval_scores[key]:.4f}")

cm_manual = confusion_matrix(a_test, a_pred, labels=[-1, 1])
print("Confusion Matrix:")
print(cm_manual)

os.makedirs("Output_Evidence", exist_ok=True)

plt.figure(figsize=(5, 5))
sns.heatmap(cm_manual, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Manual SVM (SGD)")
plt.savefig("Output_Evidence/ass3_confusion_matrix.png", bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(svm_model.loss_history)
plt.title("Hinge Loss Curve (SGD)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("Output_Evidence/ass3_loss_curve.png", bbox_inches="tight")
plt.show()
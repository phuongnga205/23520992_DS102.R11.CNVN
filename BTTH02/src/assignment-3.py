import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

train_images = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")

test_images = idx2numpy.convert_from_file("data/t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels-idx1-ubyte")

N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1) / 255.0
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) / 255.0

train_mask = (train_labels == 0) | (train_labels == 1)
test_mask = (test_labels == 0) | (test_labels == 1)

train_images_bin = train_images[train_mask]
train_labels_bin = train_labels[train_mask]
test_images_bin = test_images[test_mask]
test_labels_bin = test_labels[test_mask]

model_bin = LogisticRegression(max_iter=100)
model_bin.fit(train_images_bin, train_labels_bin)
pred_bin = model_bin.predict(test_images_bin)

metrics_bin = {
    "precision": precision_score(test_labels_bin, pred_bin),
    "recall": recall_score(test_labels_bin, pred_bin),
    "f1": f1_score(test_labels_bin, pred_bin)
}

print("="*10 + " LOGISTIC REGRESSION (SKLEARN - 2 CLASSES) " + "="*10)
for metric in metrics_bin:
    print(f"{metric}:\t{metrics_bin[metric]}")

model_multi = LogisticRegression(max_iter=100)
model_multi.fit(train_images, train_labels)
pred_multi = model_multi.predict(test_images)

metrics_multi = {
    "precision": precision_score(test_labels, pred_multi, average="macro"),
    "recall": recall_score(test_labels, pred_multi, average="macro"),
    "f1": f1_score(test_labels, pred_multi, average="macro")
}

print("\n" + "="*10 + " SOFTMAX REGRESSION (SKLEARN - 10 CLASSES) " + "="*10)
for metric in metrics_multi:
    print(f"{metric}:\t{metrics_multi[metric]}")
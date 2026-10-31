import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.logistic_regression import LogisticsRegression

np.random.seed(42)

df = pd.read_csv("../BTTH01/data_bds.csv")
df = df.dropna(subset=["price", "area", "category"])
df = df[(df["price"] > 0) & (df["area"] > 0)]

df["price_per_sqm"] = df["price"] / df["area"]

target_category = df["category"].mode()[0]
df["label"] = (df["category"] == target_category).astype(int)

df_0 = df[df["label"] == 0]
df_1 = df[df["label"] == 1]
min_count = min(len(df_0), len(df_1))

df_0_sampled = df_0.sample(n=min_count, random_state=42)
df_1_sampled = df_1.sample(n=min_count, random_state=42)
df_balanced = pd.concat([df_0_sampled, df_1_sampled])

X = df_balanced[["price", "area", "price_per_sqm"]].values
y = df_balanced["label"].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-15)

bias = np.ones((X.shape[0], 1))
X = np.concatenate([X, bias], axis=1)

num_samples = len(X)
train_size = int(num_samples * 0.8)
indices = np.random.permutation(num_samples)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

model = LogisticsRegression(epoch=5000, lr=0.1)
model.fit(X_train, y_train)

predicted_labels = model.predict(X_test)
metrics = model.evaluate(y_test, predicted_labels)

print("=" * 10)
for metric in metrics:
    print(f"{metric}:\t{metrics[metric]}")

predicted_classes = (predicted_labels >= 0.5).astype(int)
cfm = confusion_matrix(y_test, predicted_classes)
print(cfm)

plt.figure(figsize=(5, 5))
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure()
plt.plot(model.losses)
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class LinearSoftMarginSVM:
    def __init__(self, penalty_c=1.0, learning_rate=1e-6, num_epochs=30):
        self.penalty_c = penalty_c
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def compute_hinge_loss(self, X_mat, a_vec):
        decision_vals = np.dot(X_mat, self.weights) + self.bias
        hinge_terms = np.maximum(0.0, 1.0 - a_vec * decision_vals)
        reg_term = 0.5 * np.sum(self.weights ** 2)
        total_loss = reg_term + self.penalty_c * np.sum(hinge_terms)
        return total_loss

    def train(self, X_mat, a_vec):
        num_samples, num_features = X_mat.shape
        self.weights = np.zeros(num_features, dtype=np.float64)
        self.bias = 0.0
        self.loss_history = []

        for ep in tqdm(range(self.num_epochs), desc="Training Soft-Margin SVM"):
            sample_order = np.arange(num_samples)
            np.random.shuffle(sample_order)

            for idx in sample_order:
                x_i = X_mat[idx]
                a_i = a_vec[idx]
                y_i = np.dot(x_i, self.weights) + self.bias

                if a_i * y_i >= 1.0:
                    dw = self.weights
                    db = 0.0
                else:
                    dw = self.weights - self.penalty_c * a_i * x_i
                    db = -self.penalty_c * a_i

                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db

            epoch_loss = self.compute_hinge_loss(X_mat, a_vec)
            self.loss_history.append(epoch_loss)

    def predict(self, X_mat):
        raw_outputs = np.dot(X_mat, self.weights) + self.bias
        predicted_labels = np.where(raw_outputs >= 0.0, 1, -1)
        return predicted_labels

    def compute_metrics(self, a_true, a_pred):
        prec = precision_score(a_true, a_pred, pos_label=1, zero_division=0)
        rec = recall_score(a_true, a_pred, pos_label=1, zero_division=0)
        f1 = f1_score(a_true, a_pred, pos_label=1, zero_division=0)
        return {
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
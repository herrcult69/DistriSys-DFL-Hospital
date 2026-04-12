import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Basic logistic regression
# --------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class SimpleLogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def train_local(self, X, y, epochs=5, lr=0.01):
        n = len(X)
        for _ in range(epochs):
            y_hat = self.predict_proba(X)

            dw = (1 / n) * np.dot(X.T, (y_hat - y))
            db = (1 / n) * np.sum(y_hat - y)

            self.w -= lr * dw
            self.b -= lr * db

    def get_params(self):
        return self.w.copy(), self.b

    def set_params(self, w, b):
        self.w = w.copy()
        self.b = b

# --------------------------
# 2. Load easy dataset
# --------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------
# 3. Split training data into 3 hospitals
# --------------------------
n = len(X_train)
idx1 = int(0.4 * n)
idx2 = int(0.7 * n)

X_h1, y_h1 = X_train[:idx1], y_train[:idx1]
X_h2, y_h2 = X_train[idx1:idx2], y_train[idx1:idx2]
X_h3, y_h3 = X_train[idx2:], y_train[idx2:]

hospitals = [
    {"name": "Hospital A", "X": X_h1, "y": y_h1},
    {"name": "Hospital B", "X": X_h2, "y": y_h2},
    {"name": "Hospital C", "X": X_h3, "y": y_h3},
]

# --------------------------
# 4. Create one model per hospital
# --------------------------
n_features = X_train.shape[1]
models = [SimpleLogisticRegression(n_features) for _ in range(3)]

# --------------------------
# 5. Federated rounds
# --------------------------
rounds = 10
local_epochs = 5
lr = 0.01

for r in range(rounds):
    # local training
    for i in range(3):
        models[i].train_local(hospitals[i]["X"], hospitals[i]["y"], epochs=local_epochs, lr=lr)

    # decentralized-style averaging (simple full average here)
    weights = []
    biases = []
    for model in models:
        w, b = model.get_params()
        weights.append(w)
        biases.append(b)

    avg_w = np.mean(weights, axis=0)
    avg_b = np.mean(biases)

    # send averaged params back to all hospitals
    for model in models:
        model.set_params(avg_w, avg_b)

    # quick progress print
    preds = models[0].predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"Round {r+1}: Test Accuracy = {acc:.4f}")

# --------------------------
# 6. Final evaluation
# --------------------------
final_preds = models[0].predict(X_test)
final_acc = np.mean(final_preds == y_test)

print("\nFinal Accuracy:", round(final_acc, 4))

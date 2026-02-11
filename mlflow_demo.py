
import os
os.environ["MLFLOW_ENABLE_ARTIFACTS_DESTINATION_LOGGING"] = "false"

import pennylane as qml
from pennylane import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:///C:/Users/a1/Desktop/pennylane/mlruns")

# Load breast cancer dataset from scikit-learn
bc = load_breast_cancer()
X = bc.data
y = bc.target

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
mlflow.set_experiment("pennylane_vs_classical")

# Classical ML model
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

with mlflow.start_run(run_name="classical_logreg"):
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    mlflow.sklearn.log_model(clf, "model")
    # Save confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Classical Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("classical_cm.png")
    mlflow.log_artifact("classical_cm.png")
    plt.close()
    print(f"Classical Logistic Regression accuracy: {acc}")

# --- Variational Quantum Classifier ---
import pennylane.optimize as optimize

# Use PCA to reduce features for quantum classifier
from sklearn.decomposition import PCA
n_qubits = 4
pca = PCA(n_components=n_qubits)
X_train_q = pca.fit_transform(X_train)
X_test_q = pca.transform(X_test)

# Quantum circuit with more layers/qubits
def variational_circuit(x, weights):
    # Feature encoding
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)
    # Variational layers
    for l in range(weights.shape[0]):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(x, weights):
    variational_circuit(x, weights)
    return qml.expval(qml.PauliZ(0))

def quantum_predict(X, weights):
    preds = [qnode(x, weights) for x in X]
    return np.array([1 if p > 0 else 0 for p in preds])

def cost(weights, X, y):
    y_pred = quantum_predict(X, weights)
    return np.mean(y_pred != y)

# Initialize weights for more layers/qubits
np.random.seed(42)
n_layers = 3
weights = np.random.randn(n_layers, n_qubits, requires_grad=True)
opt = optimize.AdamOptimizer(stepsize=0.1)
epochs = 40
for i in range(epochs):
    weights = opt.step(lambda w: cost(w, X_train_q, y_train), weights)
    if (i+1) % 10 == 0:
        print(f"Epoch {i+1}: train error = {cost(weights, X_train_q, y_train):.3f}")

# Evaluate quantum classifier and log to MLflow
y_pred_q = quantum_predict(X_test_q, weights)
acc_q = accuracy_score(y_test, y_pred_q)
print(f"Quantum classifier accuracy: {acc_q}")

with mlflow.start_run(run_name="quantum_classifier", nested=True):
    mlflow.log_param("circuit", f"{n_qubits}-qubit, {n_layers}-layer VQC")
    mlflow.log_metric("accuracy", acc_q)
    mlflow.log_metric("precision", precision_score(y_test, y_pred_q))
    mlflow.log_metric("recall", recall_score(y_test, y_pred_q))
    mlflow.log_metric("f1", f1_score(y_test, y_pred_q))
    # Save confusion matrix as artifact
    cm_q = confusion_matrix(y_test, y_pred_q)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm_q, annot=True, fmt="d", cmap="Blues")
    plt.title("Quantum Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("quantum_cm.png")
    mlflow.log_artifact("quantum_cm.png")
    plt.close()

# --- MLflow UI instructions ---
if __name__ == "__main__":
    print("\n--- MLflow Tracking ---")
    print("To view experiment results, run this command in your terminal:")
    print(f"mlflow ui --backend-store-uri {os.path.abspath('mlruns')}")
    print("Then open http://localhost:5000 in your browser.")

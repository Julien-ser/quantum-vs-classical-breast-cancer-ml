# Quantum vs Classical ML: Breast Cancer Classification

This project benchmarks classical and quantum machine learning models on the breast cancer dataset using PennyLane, scikit-learn, and MLflow.

## Setup Instructions

1. **Activate the virtual environment**:
   ```powershell
   .\penlane\Scripts\activate
   ```
2. **Install dependencies**:
   ```powershell
   .\penlane\Scripts\pip install pennylane scikit-learn mlflow matplotlib seaborn
   ```
3. **Run the experiment**:
   ```powershell
   python mlflow_demo.py
   ```
4. **View results in MLflow UI**:
   ```powershell
   mlflow ui --backend-store-uri file:///C:/Users/a1/Desktop/pennylane/mlruns
   ```
   Open http://localhost:5000 in your browser.

## Results (Sample Output)

```
Classical Logistic Regression accuracy: 0.97
Quantum classifier accuracy: 0.65
```

### Classical Model Confusion Matrix
![Classical Confusion Matrix](classical_cm.png)

### Quantum Model Confusion Matrix
![Quantum Confusion Matrix](quantum_cm.png)

## Project Structure
- `main.py`: Simple PennyLane circuit demo
- `mlflow_demo.py`: Full classical vs quantum ML experiment
- `classical_cm.png`, `quantum_cm.png`: Confusion matrices
- `mlruns/`: MLflow experiment tracking

## Summary
- Classical model achieves high accuracy (Logistic Regression)
- Quantum model (variational circuit, 4 qubits, 3 layers, PCA features) achieves moderate accuracy
- All metrics and confusion matrices are tracked in MLflow for comparison

---
For more PennyLane tutorials and documentation, visit: https://pennylane.ai/

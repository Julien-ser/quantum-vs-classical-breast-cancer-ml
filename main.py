import pennylane as qml
from pennylane import numpy as np

# Define a device with 1 qubit
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))

if __name__ == "__main__":
    x = np.pi / 4
    result = circuit(x)
    print(f"Expectation value for RX({x}): {result}")

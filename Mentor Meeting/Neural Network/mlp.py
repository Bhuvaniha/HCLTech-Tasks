
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1.0 - a)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5   
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   
b2 = np.zeros((1, 1))

lr = 1.0
epochs = 5000
n = X.shape[0]          
loss_history = []

for epoch in range(epochs):

    Z1 = X @ W1 + b1            # weighted sum of hidden layer
    A1 = sigmoid(Z1)            # hidden activation
    Z2 = A1 @ W2 + b2           # weighted sum of output layer
    A2 = sigmoid(Z2)            # predicted output

    loss = np.mean((y - A2) ** 2)
    loss_history.append(loss)

    # output layer gradients
    dZ2 = 2.0 * (A2 - y) / n * sigmoid_derivative(A2)   # dL/dZ2
    dW2 = A1.T @ dZ2                                     # dL/dW2
    db2 = np.sum(dZ2, axis=0, keepdims=True)             # dL/db2

    # hidden layer gradients (error is pushed back through W2)
    dZ1 = (dZ2 @ W2.T) * sigmoid_derivative(A1)          # dL/dZ1
    dW1 = X.T @ dZ1                                      # dL/dW1
    db1 = np.sum(dZ1, axis=0, keepdims=True)             # dL/db1

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 500 == 0 or epoch == epochs - 1:
        print("  Epoch %5d | loss = %.6f" % (epoch, loss))

print("\nFINAL PREDICTIONS:")
print("  x1  x2  | target | output  | rounded")
for i in range(n):
    rounded = 1 if A2[i, 0] >= 0.5 else 0
    print("   %d   %d  |   %d    | %.4f   |   %d"
          % (X[i, 0], X[i, 1], y[i, 0], A2[i, 0], rounded))

correct = sum(1 for i in range(n) if (A2[i, 0] >= 0.5) == y[i, 0])
print("\nAccuracy: %d/%d" % (correct, n))

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, color='royalblue', lw=2)
plt.title('Learning Curve - XOR (MLP 2->4->1)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve_xor.png', dpi=120)
print("Saved plot: loss_curve_xor.png")
plt.show()

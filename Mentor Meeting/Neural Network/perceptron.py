import numpy as np

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
y = [0, 0, 0, 1]  

def step(z):
    return 1 if z >= 0 else 0

def forward(inputs, weights, bias):
    total = bias
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    prediction = step(total)
    return total, prediction

weights = [0.0, 0.0]
bias = 0.0
learning_rate = 0.1

print("\nBEFORE training (weights = [0.0, 0.0], bias = 0.0):")
print("\n  inputs   | prediction | actual")
for i in range(len(X)):
    total, prediction = forward(X[i], weights, bias)
    print("  %s    |     %s      |   %s" % (X[i], prediction, y[i]))

print("\nTraining...")
epochs = 100
for epoch in range(epochs):
    errors = 0
    for i in range(len(X)):
        total, prediction = forward(X[i], weights, bias)
        error = y[i] - prediction

        for w in range(len(weights)):
            weights[w] += learning_rate * error * X[i][w]
        bias += learning_rate * error

        errors += abs(error)

    if epoch % 10 == 0 or errors == 0:
        print("  Epoch %3d | errors = %d | weights = [%.2f, %.2f] | bias = %.2f"
              % (epoch, errors, weights[0], weights[1], bias))
    if errors == 0:
        break

print("\nAFTER training:")
print("  inputs   | prediction | actual")
correct = 0
for i in range(len(X)):
    total, prediction = forward(X[i], weights, bias)
    if prediction == y[i]:
        correct += 1
    print("  %s    |     %s      |   %s" % (X[i], prediction, y[i]))
print("\nAccuracy: %d/%d" % (correct, len(X)))
print("Decision rule: 1*x1 + %.2f*x2 + %.2f >= 0  ->  1" % (weights[1], bias))
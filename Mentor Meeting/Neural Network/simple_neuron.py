import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)         
    return 1.0 / (1.0 + np.exp(-x))

def step(x):
    return 1 if x >= 0 else 0

inputs = [1, 2]
weights = [0.5, 0.3]
bias = 0.1

print("A SINGLE NEURON - calculated step by step")
print("inputs  :", inputs)
print("weights :", weights)
print("bias    :", bias)

print("\nStep 1: multiply each input by its weight and add them up")
weighted_sum = 0.0
for i in range(len(inputs)):
    contribution = inputs[i] * weights[i]
    print("   %s * %s = %s" % (inputs[i], weights[i], contribution))
    weighted_sum += contribution
print("   weighted sum = %.2f" % weighted_sum)

print("\nStep 2: add the bias")
z = weighted_sum + bias
print("   %.2f + %.2f = %.2f" % (weighted_sum, bias, z))

print("\nStep 3: activation function")
output = sigmoid(z)
print("   sigmoid(%.2f) = %.4f " % (z, output))

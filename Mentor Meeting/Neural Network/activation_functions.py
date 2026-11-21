import numpy as np

# 1. STEP FUNCTION
def step(x):
    return 1 if x >= 0 else 0

def step_derivative(x):
    return 0

# 2. SIGMOID
def sigmoid(x):
    x = np.clip(x, -500, 500)      
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# 3. ReLU  (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 4. TANH
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

# 5. SOFTMAX
def softmax(x):
    exp_values = np.exp(x)               
    return exp_values / np.sum(exp_values)

#                        DEMONSTRATIONS
print("ACTIVATION FUNCTIONS - small examples")

sample_inputs = [-3.0, -1.0, 0.0, 1.0, 3.0]

# --- 1. step ---
print("\n1) STEP FUNCTION")
for v in sample_inputs:
    print("   step(%5s) = %s" % (v, step(v)))
print("   derivative is always 0 -> cannot learn (see comment above).")

# --- 2. sigmoid ---
print("\n2) SIGMOID")
for v in sample_inputs:
    print("   sigmoid(%5s) = %.4f" % (v, sigmoid(v)))
print("   derivative at 0 = %.4f  (its steepest point)" % sigmoid_derivative(0))

# --- 3. relu ---
print("\n3) ReLU")
for v in sample_inputs:
    print("   relu(%5s) = %.4f" % (v, relu(v)))
print("   derivative at -2 = %s , at +2 = %s" % (relu_derivative(-2), relu_derivative(2)))

# --- 4. tanh ---
print("\n4) TANH")
for v in sample_inputs:
    print("   tanh(%5s) = %.4f" % (v, tanh(v)))
print("   derivative at 0 = %.4f  (its steepest point)" % tanh_derivative(0))

# --- 5. softmax ---
print("\n5) SOFTMAX")
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("   input scores       : %s" % scores)
print("   softmax output     : %s" % np.round(probs, 4))
print("   sum of the outputs : %.4f   <- always 1.0" % np.sum(probs))


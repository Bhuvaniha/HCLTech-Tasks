
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


print("Creating sequence data...")
X = []
y = []

for i in range(0, 18):
    seq = [i, i+1, i+2]
    target = i+3
    X.append(seq)
    y.append(target)

X = np.array(X)
y = np.array(y)

print(f"Input shape: {X.shape}")  # (samples, 3)
print(f"Target shape: {y.shape}")
print("\nTraining data (first 5):")
for i in range(min(5, len(X))):
    print(f"  {X[i]} -> {y[i]}")

X_max = X.max()
y_max = y.max()
X_norm = X.astype('float32') / X_max
y_norm = y.astype('float32') / y_max

X_norm = X_norm.reshape(X_norm.shape[0], X_norm.shape[1], 1)
print(f"\nReshaped input for RNN: {X_norm.shape}")  # (samples, 3, 1)

print("\nBuilding SimpleRNN model...")
model = Sequential()

model.add(SimpleRNN(units=32, activation='tanh', input_shape=(3, 1)))

model.add(Dense(units=1))

model.summary()

model.compile(optimizer='adam', loss='mse')
print("\nTraining RNN (this may take a moment)...")
model.fit(X_norm, y_norm, epochs=500, batch_size=4, verbose=0)
print("Training complete!")

print("\nTesting predictions:")
test_sequences = [[5,6,7], [6,7,8], [10,11,12]]

for seq in test_sequences:
    test_norm = np.array(seq).reshape(1, 3, 1).astype('float32') / X_max
    pred_norm = model.predict(test_norm, verbose=0)
    pred = pred_norm[0][0] * y_max
    expected = seq[-1] + 1
    print(f"  {seq} -> predicted: {pred:.2f} (expected ~{expected})")

print("\nOn training examples:")
for seq in [[0,1,2], [1,2,3], [4,5,6]]:
    test_norm = np.array(seq).reshape(1, 3, 1).astype('float32') / X_max
    pred_norm = model.predict(test_norm, verbose=0)
    pred = pred_norm[0][0] * y_max
    expected = seq[-1] + 1
    print(f"  {seq} -> predicted: {pred:.2f} (expected ~{expected})")


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. LOAD MNIST DATASET (handwritten digits 0-9)
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 2. PREPROCESS THE DATA
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. BUILD A SMALL CNN MODEL
# Following the flow: Input -> Convolution -> ReLU -> Pooling -> Flatten -> Dense -> Output
print("\nBuilding CNN model...")
model = Sequential()

# CONVOLUTION LAYER 1: Detect basic features like edges
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

# POOLING LAYER 1
model.add(MaxPooling2D(pool_size=(2,2)))

# CONVOLUTION LAYER 2: Detect more complex patterns
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

# POOLING LAYER 2
model.add(MaxPooling2D(pool_size=(2,2)))

# FLATTEN: Convert 2D feature maps to 1D
model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

# OUTPUT LAYER
model.add(Dense(units=10, activation='softmax'))

# View model architecture
model.summary()

# 4. COMPILE THE MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. TRAIN THE MODEL (few epochs to run quickly)
print("\nTraining CNN (this may take a moment)...")
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

# 6. EVALUATE ON TEST DATA
print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 7. PREDICT ON A SAMPLE IMAGE
print("\nMaking prediction on a sample image...")
sample_index = 0
sample_image = X_test[sample_index]
# Reshape for prediction (model expects batch dimension)
sample_image_batch = sample_image.reshape(1, 28, 28, 1)

# Get prediction probabilities
prediction = model.predict(sample_image_batch, verbose=0)
predicted_digit = np.argmax(prediction)
true_digit = np.argmax(y_test[sample_index])

print(f"True Digit: {true_digit}")
print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence: {prediction[0][predicted_digit] * 100:.2f}%")

# 8. VISUALIZE
plt.figure(figsize=(6,6))
plt.imshow(sample_image.reshape(28,28), cmap='gray')
plt.title(f'True: {true_digit} | Predicted: {predicted_digit}')
plt.axis('off')
plt.show()


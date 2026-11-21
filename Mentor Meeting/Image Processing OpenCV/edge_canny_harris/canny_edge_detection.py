import cv2
import matplotlib.pyplot as plt
import os

# Load the image
image_path = "Moon.jpeg"
if not os.path.exists(image_path):
    image_path = "moon.jpeg"
if not os.path.exists(image_path):
    image_path = "/content/Moon.jpeg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not load image from {image_path}. Put Moon.jpeg in this folder or update image_path.")

# Apply Canny edge detector
edges = cv2.Canny(image, 100, 200)

# Display the original image and the edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Edges')
plt.imshow(edges, cmap='gray')
plt.show()

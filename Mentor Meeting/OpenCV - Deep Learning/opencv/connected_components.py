
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Creating a sample binary image with separate objects...")

image = np.ones((200, 300), dtype=np.uint8) * 255

cv2.rectangle(image, (20, 30), (70, 50), 0, -1)  # filled rectangle
cv2.rectangle(image, (150, 30), (200, 50), 0, -1)
cv2.rectangle(image, (80, 70), (130, 90), 0, -1)
cv2.rectangle(image, (80, 70), (100, 110), 0, -1)
cv2.rectangle(image, (30, 130), (100, 150), 0, -1)
cv2.rectangle(image, (200, 140), (230, 170), 0, -1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Binary Image')
plt.axis('off')

gray = image.copy()
print("Image is grayscale.")

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
print("Applied thresholding and inverted (objects are now white)")

plt.subplot(1, 2, 2)
plt.imshow(binary, cmap='gray')
plt.title('Thresholded Binary (Inverted)')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nFinding connected components...")
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

print(f"Number of connected components found: {num_labels}")
print("(Note: Label 0 is the background)")

output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

print("\nComponent Details:")
print("-" * 50)
for i in range(1, num_labels):  # Skip label 0 (background)
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    cx, cy = centroids[i]
    
    print(f"Component {i}:")
    print(f"  - Area: {area} pixels")
    print(f"  - Bounding Box: x={x}, y={y}, width={w}, height={h}")
    print(f"  - Centroid: ({cx:.1f}, {cy:.1f})")
    
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.putText(output, f"C{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print("-" * 50)
print(f"Total objects (excluding background): {num_labels - 1}")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Connected Components with Bounding Boxes')
plt.axis('off')
plt.show()


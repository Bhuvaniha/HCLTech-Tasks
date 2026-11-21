
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Loading image...")
image_path = '../images/sample.png'

try:
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Image not found")
    print(f"Loaded {image_path}")
except:
    print("Creating a sample image for demonstration...")
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
    cv2.circle(img, (300, 100), 60, (0, 255, 0), -1)
    cv2.line(img, (50, 250), (350, 250), (255, 0, 0), 5)
    cv2.rectangle(img, (200, 200), (280, 280), (0, 0, 0), 3)
    cv2.putText(img, 'ORB', (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Converted to grayscale")

print("\nCreating ORB detector...")
orb = cv2.ORB_create(nfeatures=500)  # Detect up to 500 keypoints

print("Detecting keypoints and computing descriptors...")
keypoints, descriptors = orb.detectAndCompute(gray, None)

print(f"\nNumber of keypoints detected: {len(keypoints)}")
if descriptors is not None:
    print(f"Descriptor shape: {descriptors.shape}")
    print(f"(Each descriptor is {descriptors.shape[1]} bytes for ORB binary descriptor)")
else:
    print("No descriptors found")

img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("\nGreen circles show keypoints (position, scale, orientation)")

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.axis('off')
plt.show()

print("\n" + "="*50)
print("FEATURE MATCHING DEMONSTRATION")
print("="*50)

print("Creating a transformed version of image for matching...")
rows, cols = gray.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 0.9)  # 10 degrees rotation, 0.9 scale
gray2 = cv2.warpAffine(gray, M, (cols, rows))

keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
print(f"Keypoints in image 2: {len(keypoints2)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

if descriptors is not None and descriptors2 is not None:
    print("Matching features...")
    matches = bf.match(descriptors, descriptors2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"Number of matches found: {len(matches)}")
    print(f"Best match distance: {matches[0].distance:.2f}")
    print(f"Worst match distance: {matches[-1].distance:.2f}")
    
    img_matches = cv2.drawMatches(img, keypoints, cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR), 
                                   keypoints2, matches[:20], None, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching (Top 20 matches)')
    plt.axis('off')
    plt.show()
else:
    print("Could not match - descriptors missing")


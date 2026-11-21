# Deep Learning & OpenCV Beginner Lab

A beginner-friendly educational project to learn CNN, RNN, LSTM, GRU, Connected Component Analysis, and Feature Descriptors with simple, well-commented code.

## Learning Path

```
Perceptron
   ↓
Neural Network
   ↓
Deep Neural Network
   ↓
CNN → Images
RNN → Sequences
LSTM → Long-term sequence memory
GRU → Simplified gated sequence memory
```

```
OpenCV
   ↓
Image Processing
   ↓
Connected Components
   ↓
Keypoints
   ↓
Feature Descriptors
```

## Project Structure

```text
deep_learning_opencv_demo/
│
├── cnn/
│   └── cnn_demo.py
├── rnn/
│   └── rnn_demo.py
├── lstm/
│   └── lstm_demo.py
├── gru/
│   └── gru_demo.py
├── opencv/
│   ├── connected_components.py
│   └── feature_descriptors.py
├── images/
│   └── sample.png
├── requirements.txt
└── README.md
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Packages used:
- `tensorflow` - Deep learning (CNN, RNN, LSTM, GRU)
- `keras` - High-level API (comes with TensorFlow)
- `opencv-python` - Computer vision
- `numpy` - Numerical operations
- `matplotlib` - Visualization

## How to Run

Run each demo independently:

```bash
python cnn/cnn_demo.py
python rnn/rnn_demo.py
python lstm/lstm_demo.py
python gru/gru_demo.py
python opencv/connected_components.py
python opencv/feature_descriptors.py
```

---

## 1. CNN (Convolutional Neural Network)

### What is it?
A CNN is a type of neural network designed specifically for image data. It uses convolutional layers to automatically learn spatial features.

### Why do we use it?
- Preserves spatial relationships between pixels
- Uses fewer parameters than fully connected networks
- Automatically learns features (no manual feature engineering)
- Great for image classification, object detection

### Simple real-world example
Facial recognition on your phone - CNN learns to detect eyes, nose, face structure.

### Basic working principle
```
Input Image → Convolution → ReLU → Pooling → Flatten → Dense Layer → Output
```

### Code: `cnn/cnn_demo.py`
Uses MNIST handwritten digits dataset.

**Key concepts explained in code:**
- **Convolution**: Sliding a kernel/filter over image to detect patterns
- **Kernel/Filter**: Small weight matrix that detects specific features
- **Feature Maps**: Output showing where patterns are detected
- **Pooling (MaxPooling)**: Reduces size, keeps important features
- **Flatten**: Converts 2D to 1D for Dense layers
- **Difference from FCN**: CNNs respect spatial structure

**Simple analogy:** *"A CNN learns to detect simple patterns like edges first, then combines them into more complex patterns."*

### Expected output
- Model summary
- Training accuracy per epoch
- Test accuracy (e.g., ~98%)
- Sample image with true vs predicted digit

### Important functions/classes
- `Conv2D` - Convolutional layer
- `MaxPooling2D` - Pooling layer
- `Flatten` - Flatten layer
- `Dense` - Fully connected layer
- `mnist.load_data()` - Load dataset

### How the code works
1. Load and preprocess MNIST (reshape, normalize, one-hot encode)
2. Build small CNN with Conv2D, MaxPooling2D, Flatten, Dense
3. Compile with adam optimizer and categorical crossentropy
4. Train for 3 epochs
5. Evaluate and predict on sample image

---

## 2. RNN (Recurrent Neural Network)

### What is it?
An RNN is a neural network with memory. It processes sequential data by passing information from one time step to the next.

### Why do we use it?
- Works with sequences where order matters
- Has memory of previous inputs via hidden state
- Good for time series, text, speech

### Simple real-world example
Predicting next word in your phone's keyboard - it remembers previous words.

### Basic working principle
```
x1 → RNN → h1
          ↓
x2 → RNN → h2
          ↓
x3 → RNN → h3
          ↓
        output
```

### Code: `rnn/rnn_demo.py`
Simple sequence prediction: `[1,2,3] → 4`, `[2,3,4] → 5` etc (learns next number = previous+1)

### Key concepts
- **Sequential data**: Order matters
- **Hidden state (h)**: RNN's memory carried across time steps
- **Information flow**: Current input + previous hidden state → new hidden state + output
- **Why FCN can't remember**: Feed-forward processes each input independently with no loops/memory

### Expected output
- Predictions like `1 2 3 -> 4.00`

### Important functions/classes
- `SimpleRNN` - Basic recurrent layer
- `Dense` - Output layer

### How the code works
1. Create sequence pairs where target is next number
2. Reshape to (samples, timesteps, features)
3. Build SimpleRNN with 10 units
4. Train for 200 epochs
5. Test predictions

---

## 3. LSTM (Long Short-Term Memory)

### What is it?
LSTM is an improved RNN with gates that control what to remember/forget. Solves vanishing gradient problem.

### Why do we use it?
- Remembers information over long sequences
- Better at long-term dependencies
- More stable training than simple RNN

### Simple real-world example
Language translation - needs to remember context from earlier in sentence.

### Basic working principle - The Three Gates
```
Forget Gate  → What old information should I forget?
Input Gate   → What new information should I remember?  
Output Gate  → What information should I output?
```

LSTM introduced to solve **vanishing gradients** and **remembering over longer sequences**.

### Code: `lstm/lstm_demo.py`
Same sequence task as RNN for easy comparison.

### Expected output
- Similar predictions to RNN on short sequences
- Demonstrates controlled memory via gates

### Important functions/classes
- `LSTM` - LSTM layer with forget/input/output gates
- `Dense` - Output layer

### How the code works
Same structure as RNN, but uses `LSTM(units=10)` instead of `SimpleRNN`. Gates are handled internally.

---

## 4. GRU (Gated Recurrent Unit)

### What is it?
GRU is a simplified version of LSTM with fewer gates. Combines cell state and hidden state.

### Why do we use it?
- Simpler and faster than LSTM
- Fewer parameters
- Often performs similarly to LSTM
- Good for smaller datasets

### Simple real-world example
Text summarization on resource-limited devices.

### Basic working principle - The Two Gates
```
Update Gate → How much of past information to keep vs forget? (combines forget+input)
Reset Gate  → How much of past information to ignore?
```

### Code: `gru/gru_demo.py`
Same sequence task + comparison table.

### Comparison Table
| Model | Memory | Gates | Complexity |
|------|--------|-------|------------|
| RNN  | Basic  | None  | Low        |
| LSTM | Long-term | 3 (Forget, Input, Output) | Higher |
| GRU  | Long-term | 2 (Update, Reset) | Medium     |

### Expected output
- Predictions + comparison table

### Important functions/classes
- `GRU` - GRU layer with update/reset gates
- `Dense` - Output layer

### How the code works
Uses `GRU(units=10)` - merges cell/hidden state, uses only 2 gates internally.

---

## 5. Connected Component Analysis

### What is it?
Connected Component Analysis identifies groups of connected foreground pixels as separate objects. Each connected group gets a unique label.

### Why do we use it?
- Count separate objects in an image
- Measure properties (area, centroid, bounding box)
- Separate touching/blended regions

### Simple real-world example
Counting coins on a table from a binary image, or counting cells in microscope image.

### Difference: Connected Component Analysis vs Contour Detection
| Aspect | Connected Component Analysis | Contour Detection |
|--------|-------------------------------|-------------------|
| Purpose | Label entire connected regions | Find boundaries/edges of objects |
| Output | Labels, area, centroid, stats | List of points forming outline |
| Best for | Counting objects, measuring area | Shape analysis, object outline |
| Method | Groups connected pixels | Traces boundary curves |

### Code: `opencv/connected_components.py`

### Steps in code
1. Create binary image with separate objects
2. Convert to grayscale
3. Threshold and invert (objects become white foreground)
4. Find components with `cv2.connectedComponentsWithStats()`
5. Calculate component number, bounding box, area, centroid
6. Draw bounding boxes
7. Display results

### Expected output
- Component details printed (area, bounding box, centroid)
- Images showing original, thresholded, and labeled components

### Important functions
- `cv2.connectedComponentsWithStats()` - Find components with statistics
- `cv2.threshold()` - Binarize image
- `cv2.rectangle()` - Draw bounding boxes
- `cv2.putText()` - Label components

### How the code works
- Label 0 = background
- Labels 1,2,... = separate objects
- Stats gives bounding box and area for each
- Centroids give center points

---

## 6. Feature Descriptors (ORB)

### What is it?
Feature detection finds interesting points (keypoints). Feature description creates a numerical "fingerprint" (descriptor) of that area for matching.

### Keypoint vs Descriptor
- **Keypoint** → *WHERE* is an interesting feature? (corners, edges, patterns)
- **Descriptor** → *WHAT* does that feature look like? (numerical representation)

### Why do we use it?
- Find same object in different images (rotation, scale changes)
- Image stitching, object recognition
- Panorama creation

### Simple real-world example
Google Photos recognizing same person in different photos, or panorama stitching.

### Why ORB?
- Free to use (SIFT/SURF have patents)
- Fast and efficient
- Great for beginners
- Detects keypoints + computes descriptors

### Code: `opencv/feature_descriptors.py`

### Steps in code
1. Load image
2. Convert to grayscale
3. Create ORB detector
4. Detect keypoints and compute descriptors with `orb.detectAndCompute()`
5. Draw keypoints on image
6. Print number of keypoints and descriptor shape
7. Feature matching with `cv2.BFMatcher` between original and transformed image

### Expected output
- Image with green circles showing keypoints (position/scale/orientation)
- Matching lines between keypoints in two images
- Print: number of keypoints, descriptor shape, match counts/distances

### Important functions
- `cv2.ORB_create()` - Create ORB detector
- `orb.detectAndCompute()` - Detect keypoints and compute descriptors
- `cv2.drawKeypoints()` - Visualize keypoints
- `cv2.BFMatcher()` - Brute force matching
- `cv2.drawMatches()` - Visualize matches

### How the code works
- Each keypoint marks a distinctive location
- Descriptor is binary vector describing local neighborhood
- BFMatcher finds closest descriptors by Hamming distance (for binary)
- Lower distance = better match
- crossCheck ensures mutual best matches

---

## Tips for Learning
- Run each file separately and read the comments
- Change parameters and see what happens
- For CNN, try different filters/epochs
- For sequences, try longer sequences
- For OpenCV, experiment with different images

## Notes
- This is for **learning concepts**, not production
- Uses small models/datasets for speed
- All code is beginner-friendly with extensive comments

# Image Processing with OpenCV — Beginner Project

A small, readable Python project that demonstrates four classical image-processing
topics: **filtering**, **edge detection**, **morphological operations** and
**thresholding**. Everything is split into separate modules so each idea can be
explained (and tested) on its own.

---

## 1. Project Overview

| | |
|---|---|
| **Goal** | Learn and clearly explain basic OpenCV techniques to a mentor |
| **Language** | Python 3 |
| **Libraries** | `opencv-python` (cv2) and `numpy` only — no ML, no frameworks |
| **Input** | `input/image.jpg` |
| **Output** | 14 processed images in `output/` |
| **Concepts** | Filtering, Edge Detection, Morphology, Thresholding, Modular coding |

The program loads one image and produces:

```
output/02_grayscale.jpg          09_global_threshold.jpg
output/03_gaussian_blur.jpg      10_otsu_threshold.jpg
output/04_median_blur.jpg        11_adaptive_threshold.jpg
output/05_average_blur.jpg       12_erosion.jpg
output/06_sobel_edges.jpg        13_dilation.jpg
output/07_canny_edges.jpg        14_opening.jpg
output/08_laplacian_edges.jpg    15_closing.jpg
```

---

## 2. What is image processing?

An image is just a grid of numbers:

* **Colour image** → three grids (Blue, Green, Red), shape `(height, width, 3)`
* **Greyscale image** → one grid, shape `(height, width)`, values `0` (black) to `255` (white)

Image processing is **reading those numbers, changing them according to a rule, and
writing them back**. The program shown here uses no artificial intelligence at all —
every result comes from a mathematical rule (average, derivative, threshold, kernel).

Typical uses: cleaning up camera photos, making scanned documents readable,
detecting objects, measuring shapes, preparing images for printing or OCR.

---

## 3. What is OpenCV?

**OpenCV (Open Source Computer Vision Library)** is a free, widely used library
containing hundreds of ready-made image-processing functions.

```python
import cv2                                  # the whole library
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Useful facts to mention to your mentor:

* It is written in C/C++ but has Python bindings, so it is very fast.
* It is used by most self-driving, medical-imaging and face-recognition systems.
* It is **not** an AI/ML library — classical techniques like the ones in this
  project are its foundation.
* Images are NumPy arrays, so anything you can do with arrays you can do with images.
* OpenCV reads colour images in **BGR** order, not RGB. This is the single most
  common beginner mistake.

---

## 4. Installation

```bash
# 1. Go into the project folder
cd image_processing_opencv

# 2. (recommended) create a virtual environment
python -m venv .venv

# 3. Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. Install the required packages
pip install -r requirements.txt
```

`requirements.txt` contains only:

```
opencv-python>=4.5
numpy>=1.21
```

> Note: `opencv-python` needs a desktop environment for `cv2.imshow()`. On a
> headless server install `opencv-python-headless` instead and run with `--no-show`.

---

## 5. How to run

```bash
python main.py             # process, display windows, save results
python main.py --no-show   # process and save only (no GUI needed)
```

Expected console output (abbreviated):

```
--- STEP 1: LOAD THE IMAGE ---
[OK] Loaded image: .../input/image.jpg
     Shape (height, width, channels): (480, 640, 3)

--- STEP 2: CONVERT TO GREYSCALE ---
     Grey shape (height, width): (480, 640)

--- STEP 3: FILTERING (blurring removes noise) ---
--- STEP 4: EDGE DETECTION ---
--- STEP 5b: OTSU THRESHOLDING ---
    Otsu picked the threshold value: 168
--- STEP 6: MORPHOLOGICAL OPERATIONS ---
All 14 result images are saved in: .../output
```

Windows opened: `1. Original`, `2. Grayscale`, `3. Filtering`, `4. Edge Detection`,
`5. Thresholding`, `6. Morphology`. Press any key to close them.

---

## 6. Project structure

```
image_processing_opencv/
│
├── main.py                  <- only loads, calls modules, displays, saves
├── requirements.txt
├── README.md
│
├── input/
│   └── image.jpg              <- the image we process
│
├── output/                  <- 14 generated result images
│
└── modules/
    ├── __init__.py          <- marks the folder as a Python package
    ├── filtering.py         <- Gaussian, Median, Average blur
    ├── edge_detection.py    <- Sobel, Canny, Laplacian
    ├── morphology.py        <- Erosion, Dilation, Opening, Closing
    └── thresholding.py      <- Global, Otsu, Adaptive
```

The call flow, one line per concept:

```
main.py ──imports──► modules/filtering.py     ──► apply_gaussian_blur(image)
main.py ──imports──► modules/edge_detection.py ──► canny_edge_detection(image)
main.py ──imports──► modules/morphology.py    ──► erode_image(image)
main.py ──imports──► modules/thresholding.py  ──► otsu_threshold(image)
```

---

## 7. Explanation of each concept

### 7.1 Filtering (smoothing / blurring)

**What it is.** Sliding a small window (kernel/filter) over the image and replacing
every pixel with a new value computed from the pixels inside that window.

**Why we use it.** Real images contain **noise** — random brightness specks caused
by low light, a hot camera sensor or JPEG compression. Noise confuses every later
step (edges appear where there is nothing, thresholding produces wrong masks), so
we blur first to clean the image.

**What happens to the image.** Noise dots shrink or vanish, fine details get
slightly softer, and edges stay roughly where they were.

**Average vs Gaussian vs Median**

| | Average (Box) | Gaussian | Median |
|---|---|---|---|
| Idea | average of neighbours | weighted average, centre counts most | middle value of neighbours |
| OpenCV | `cv2.blur` | `cv2.GaussianBlur` | `cv2.medianBlur` |
| Speed | fastest | fast | slower |
| Edge preservation | poor (very soft) | good | **best** |
| Best for | quick general smoothing | before edge detection (used by Canny) | salt-and-pepper noise, keeping sharp edges |
| Blur amount for same kernel | strongest | medium | mildest |

*Median filter takes the middle value, not the average, so a few black/white specks
cannot drag the result. That is why it beats the others against dot noise.*

### 7.2 Edge detection

**What an edge is.** A place where brightness changes sharply — normally the border
between an object and its background.

**Why it is useful.** Vision cares about shape, not colour. Edges shrink an image to
thin lines, which makes objects countable, measurable and cheap to process
(measuring the size of a bolt, finding the outline of a leaf, counting cells).

**Sobel** — first derivative, two kernels:

```
Sobel X = [[-1, 0, 1],     detects VERTICAL edges
           [-2, 0, 2],     (looks left vs right)
           [-1, 0, 1]]

Sobel Y = [[-1, -2, -1],   detects HORIZONTAL edges
           [ 0,  0,  0],
           [ 1,  2,  1]]   (looks up vs down)
```

Keeping X and Y separate is useful: X tells you a vertical edge is strong there,
Y tells you a horizontal one is. `cv2.addWeighted()` merges them. Output is thick
and noise-sensitive.

**Canny** — the best general-purpose detector, in five steps:

1. **Noise reduction** — Gaussian blur, so noise is not mistaken for an edge.
2. **Gradient calculation** — Sobel X/Y find how strong the brightness change is.
3. **Non-maximum suppression** — along each gradient direction keep only the
   brightest pixel and drop the weaker neighbours → fat lines become 1-pixel lines.
4. **Double threshold** — `> high_threshold` = definite edge, `< low_threshold` =
   definitely not an edge, in-between = *maybe*.
5. **Edge tracking by hysteresis** — a *maybe* pixel survives **only if** it touches
   a definite edge; otherwise it is deleted. Weak-but-connected lines are kept,
   isolated noise blobs are thrown away.

**Laplacian** — second derivative. Where the *change of change* is big, there is an
edge. One kernel covers all directions, but it is very noise sensitive and produces
very thick edges, so it is normally combined with thresholding.

### 7.3 Morphological operations

**What they are.** Operations that change the **shape** of the white (foreground)
part of a binary image.

**What a structuring element / kernel is.** A tiny shape (5×5 ellipse, square or
plus) that is slid over the image; every pixel is judged by the shape of the pixels
under this window.

```
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
```

Think of the kernel as a small ball:

* **Dilation** — the pixel turns white if **any** neighbour under the kernel is white
  (OR logic) → shape **expands**.
* **Erosion** — the pixel turns black if **any** neighbour under the kernel is black
  (AND logic) → shape **shrinks**.

| Operation | Combination | What it fixes | Real-world use |
|---|---|---|---|
| Erosion | — | shrinks white, removes thin lines and small dots | thinning text, removing specks, splitting touching objects |
| Dilation | — | grows white, joins nearby parts, fills small holes | bolding text, repairing broken letters, closing holes in a mask |
| Opening | erosion → dilation | removes small specks, keeps large-object size | cleaning a noisy scan / removing salt-and-pepper from a white mask |
| Closing | dilation → erosion | fills small holes/gaps, keeps outer size | fixing a mask with holes, connecting a cracked line |

### 7.4 Thresholding

**What it is.** Converting a greyscale image into a black & white image by asking
each pixel one question: *is my brightness above or below the threshold?*

```
value > threshold  -> 255 (white, foreground)
value <= threshold -> 0   (black, background)
```

**Why greyscale.** The threshold is a single brightness number, so the image needs a
single brightness channel. A colour image has three channels and no single
brightness, hence `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` first.

| | Global | Otsu | Adaptive |
|---|---|---|---|
| Threshold value | chosen by me | **computed by the computer** from the histogram | different for every small region, from the local average |
| OpenCV | `cv2.threshold(..., THRESH_BINARY)` | `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)` | `cv2.adaptiveThreshold(...)` |
| Speed | fastest | fast | slower (per-pixel window) |
| Handles uneven light | **no** | no | **yes** |
| Best for | even lighting, clean scans, known conditions | normal photos/documents where object and background differ clearly | shadows, uneven lighting, handwriting, phone photos of text |
| Weakness | bright half turns all white, dark half all black | single global value, so fails on very uneven light | background speckles if the block is too large |

**Global vs adaptive in one line:** global uses **one** threshold for the whole
image; adaptive uses **one per region**, so it survives shadows.

**When Otsu is useful.** Whenever the image has a clear two-group histogram (a
bright object on a dark background, or dark text on a light page) and you do not
want to guess the cut-off value. In this project's sample image, Otsu picked `168`.

> Demo detail: the sample image has a brightness gradient (dark on the left, bright
> on the right). With global thresholding the left side keeps only ~33 % white and
> loses detail, while adaptive thresholding keeps ~92 % white on both sides. That is
> the whole point of adaptive thresholding, visible in the output images.

---

## 8. Explanation of modular coding

**What it is.** Splitting a program into small files, where each file has **one job**
and exposes **functions** with clear names.

**Why we do it.**

* **Readability** — `main.py` reads like a table of contents:
  `apply_gaussian_blur(image)`, `canny_edge_detection(image)`. You understand the
  program's intent without reading the maths.
* **Reuse** — the same function works in another project with one import line:
  `from modules.filtering import apply_gaussian_blur`.
* **Testability** — you can test edge detection without loading an image, calling
  filtering, or running the display code.
* **Maintenance** — a bug in Canny only touches `edge_detection.py`; your filtering
  code stays untouched.
* **Teamwork** — different people can work on different modules at the same time.
* **Debugging** — the error message tells you the file and line immediately.

**Without modules** `main.py` would be ~250 lines mixing file I/O, blurring,
derivatives, kernels and thresholds — hard to read and hard to change.

**With modules** each file is 40–70 lines, has one topic in its docstring, and
`main.py` is only a short script of numbered steps.

```python
# main.py                          modules/edge_detection.py
from modules.edge_detection import  def canny_edge_detection(image, ...):
    canny_edge_detection           ...
edges = canny_edge_detection(gray)
```

---

## 9. Expected output

1. Six windows appear, showing the original, the greyscale copy, and one row per topic.
2. Fourteen files are written to `output/` (names listed in section 1).
3. Console prints the loaded shape, the greyscale shape, the Otsu threshold value,
   and one "Saved ->" line per image.

What you should *see*:

| Step | Expected visual result |
|---|---|
| Blurs | the black/white noise specks disappear; median keeps the edges sharpest |
| Sobel / Laplacian | thick, grey, double-sided edge outlines |
| Canny | thin (1 px), crisp, white lines on black |
| Global threshold | bright side almost fully white, dark side loses detail |
| Otsu threshold | balanced split chosen automatically |
| Adaptive threshold | objects found on both the bright and dark side |
| Erosion | white shapes visibly shrink, thin lines vanish |
| Dilation | white shapes visibly grow, nearby parts join |
| Opening | specks gone, shape size ~unchanged |
| Closing | holes inside white regions filled |

---

## 10. Comparison tables (quick reference)

### Filtering

| Technique | OpenCV function | Kernel | Strengths | Weaknesses |
|---|---|---|---|---|
| Average / Box blur | `cv2.blur` | `(k, k)` | Simple, fastest | Blurs heavily, loses edges |
| Gaussian blur | `cv2.GaussianBlur` | `(k, k)`, odd, `sigma` | Smooth, natural, Canny uses it | Slightly slower, still softens edges |
| Median blur | `cv2.medianBlur` | `k`, odd | Best vs salt-and-pepper, keeps edges | Slower, no weighted control, works best on greyscale |

### Edge detection

| Technique | OpenCV function | Based on | Output | Noise sensitivity | Best for |
|---|---|---|---|---|---|
| Sobel | `cv2.Sobel` | 1st derivative (X and Y) | thick grey edges | high | quick directional gradients, learning |
| Canny | `cv2.Canny` | 1st derivative + NMS + hysteresis | thin binary edges | low (blurs first) | most real projects, object outlines |
| Laplacian | `cv2.Laplacian` | 2nd derivative | very thick edges | very high | all directions at once, often + threshold |

### Morphology

| Operation | OpenCV function | Combination | Effect on white | Typical use |
|---|---|---|---|---|
| Erosion | `cv2.erode` | — | shrinks / thins | remove dots, thin text |
| Dilation | `cv2.dilate` | — | grows / thickens | bold text, join parts |
| Opening | `cv2.morphologyEx(..., MORPH_OPEN, k)` | erosion → dilation | removes small noise, size ≈ same | clean a noisy mask |
| Closing | `cv2.morphologyEx(..., MORPH_CLOSE, k)` | dilation → erosion | fills small holes, size ≈ same | repair holes/gaps |

### Thresholding

| Technique | OpenCV function | Who picks the value | Handles uneven light | Best for |
|---|---|---|---|---|
| Global | `cv2.threshold(..., THRESH_BINARY)` | you (e.g. 127) | no | even lighting, known scenes |
| Otsu | `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)` | OpenCV, from the histogram | no | normal photos and document scans |
| Adaptive | `cv2.adaptiveThreshold(..., blockSize, C)` | local average per region | **yes** | shadows, handwriting, uneven light |

---

## 11. Order of operations used in `main.py`

```
load image
      │
      ▼
to greyscale ─────────────────────────────┐
      │                                   │
      ▼                                   ▼
FILTERING (blur)                    EDGE DETECTION (Sobel / Canny / Laplacian)
      │                                   │
      └──────────► THRESHOLDING (global / Otsu / adaptive)
                              │  (binary image)
                              ▼
                    MORPHOLOGY (erode / dilate / open / close)
                              │
                              ▼
                      save to output/
```

Filtering comes before edge detection because Canny needs a clean image.
Thresholding comes before morphology because morphology only asks
"white or black?" — so the picture must already be binary.
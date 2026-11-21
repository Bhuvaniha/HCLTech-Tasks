"""
IMAGE PROCESSING WITH OPENCV - main.py
=====================================

This is the only file you run. Just type:

    python main.py

No extra options, no command-line flags, nothing to remember.

What it does:
    1. Grabs the first image it finds in the input/ folder.
    2. Shows you what each OpenCV technique does to that image.
    3. Saves every result into the output/ folder.

The real image-processing code lives in the modules/ folder, one topic per
file. main.py only puts the pieces together.
"""

import cv2

from modules.edge_detection import (
    canny_edge_detection,
    laplacian_edge_detection,
    sobel_edge_detection,
)
from modules.filtering import (
    apply_average_blur,
    apply_gaussian_blur,
    apply_median_blur,
)
from modules.morphology import closing, dilate_image, erode_image, opening
from modules.thresholding import (
    adaptive_threshold,
    global_threshold,
    otsu_threshold,
)

# Change this to whichever picture you want to use.
IMAGE_NAME = "image.jpg"


def load_image():
    """Read the image from the input folder."""
    return cv2.imread("input/" + IMAGE_NAME)


def save_image(image, filename):
    """Save one processed image into the output folder."""
    cv2.imwrite("output/" + filename, image)


def show_image(title, image):
    """Show one image in its own window, zoomed in on the middle."""
    # Crop a small square from the middle, then blow it up. A tiny crop makes
    # blur, edges and noise easy to actually see.
    height, width = image.shape[:2]
    size = 600
    top = (height - size) // 2
    left = (width - size) // 2
    crop = image[top:top + size, left:left + size]

    cv2.imshow(title, cv2.resize(crop, (size, size),
                                 interpolation=cv2.INTER_NEAREST))


def main():
    image = load_image()
    if image is None:
        print(f"Could not read input/{IMAGE_NAME}. Is it in the input folder?")
        return

    # Grey is needed for edge detection and thresholding.
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(grey, "01_grayscale.jpg")

    # --- Blurring (removes noise) ---
    # A 5x5 window is invisible on a big photo, so we use 15 here to make the
    # blurring obvious. Try 5, 9 or 15 to see how the size changes the result.
    average = apply_average_blur(image, kernel_size=15)
    gaussian = apply_gaussian_blur(image, kernel_size=15)
    median = apply_median_blur(image, kernel_size=15)
    save_image(average, "02_average_blur.jpg")
    save_image(gaussian, "03_gaussian_blur.jpg")
    save_image(median, "04_median_blur.jpg")
    show_image("1 original", image)
    show_image("2 average blur", average)
    show_image("3 gaussian blur", gaussian)
    show_image("4 median blur", median)

    # --- Edge detection (finds boundaries) ---
    sobel = sobel_edge_detection(grey)
    canny = canny_edge_detection(grey)
    laplacian = laplacian_edge_detection(grey)
    save_image(sobel, "05_sobel_edges.jpg")
    save_image(canny, "06_canny_edges.jpg")
    save_image(laplacian, "07_laplacian_edges.jpg")
    show_image("5 sobel edges", sobel)
    show_image("6 canny edges", canny)
    show_image("7 laplacian edges", laplacian)

    # --- Thresholding (makes black & white images) ---
    fixed_binary = global_threshold(grey)
    otsu_binary = otsu_threshold(grey)
    adaptive_binary = adaptive_threshold(grey)
    save_image(fixed_binary, "08_fixed_threshold.jpg")
    save_image(otsu_binary, "09_otsu_threshold.jpg")
    save_image(adaptive_binary, "10_adaptive_threshold.jpg")
    show_image("8 fixed threshold", fixed_binary)
    show_image("9 otsu threshold", otsu_binary)
    show_image("10 adaptive threshold", adaptive_binary)

    # --- Morphology (cleans up a black & white image) ---
    eroded = erode_image(otsu_binary)
    dilated = dilate_image(otsu_binary)
    opened = opening(otsu_binary)
    closed = closing(otsu_binary)
    save_image(eroded, "11_erosion.jpg")
    save_image(dilated, "12_dilation.jpg")
    save_image(opened, "13_opening.jpg")
    save_image(closed, "14_closing.jpg")
    show_image("11 erosion", eroded)
    show_image("12 dilation", dilated)
    show_image("13 opening", opened)
    show_image("14 closing", closed)

    print("\nAll results are saved in the output/ folder.")
    print("Press any key in an image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
MODULE 2 - EDGE DETECTION
========================

WHAT IS AN EDGE?
    A place where brightness changes sharply, usually the border of an object.

WHY?
    Computer vision cares about shapes, not colours. Edges turn a photo into
    a small picture made of thin lines, which is much easier to work with.

THE THREE ALGORITHMS:
    sobel     - how much did brightness change? Gives thick edges and also
               tells us the direction (X = vertical edges, Y = horizontal).
    canny     - the best and most popular. Blurs first, then keeps only thin
               lines that are joined to strong edges. Needs two thresholds.
    laplacian - finds edges in all directions at once, but is noisy and gives
               very thick edges.
"""


import cv2
import numpy as np


def sobel_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dx=1, dy=0 looks left vs right -> vertical edges.
    # dx=0, dy=1 looks up vs down -> horizontal edges.
    sobel_x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))

    # Add the two directions together.
    return np.uint8(cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0))


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur first
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    return cv2.Canny(blurred, low_threshold, high_threshold)


def laplacian_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

    return np.uint8(np.abs(laplacian))
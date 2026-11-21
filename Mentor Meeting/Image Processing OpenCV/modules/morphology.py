"""
MODULE 3 - MORPHOLOGICAL OPERATIONS
==================================

WHAT ARE THEY?
    "Morphology" means the study of shape. These operations change the shape
    of the white part of a black & white image: bigger, smaller, cleaner.

THE SMALL SHAPE (KERNEL)
    A tiny shape we slide over the image. Every decision is made by looking
    at the pixels underneath it, so a bigger kernel = a bigger change.
    We pass it straight to OpenCV as a size, like (3, 3) or (5, 5).

HOW EACH ONE DECIDES
    dilation - becomes white if ANY pixel under the kernel is white
               -> the shape grows.
    erosion  - becomes black if ANY pixel under the kernel is black
               -> the shape shrinks.

THE FOUR OPERATIONS
    erosion  - eats away white: removes dots and thin lines.
    dilation - adds white: joins broken parts and fills small holes.
    opening  - erosion then dilation: removes small specks.
    closing  - dilation then erosion: fills small holes and tiny gaps.

These only ask "is this pixel white or black?", so they work best on a
BINARY image. That is why we threshold first, then do morphology.
"""


import cv2


def erode_image(image, kernel=(5, 5)):
    return cv2.erode(image, kernel)


def dilate_image(image, kernel=(5, 5)):
    return cv2.dilate(image, kernel)


def opening(image, kernel=(5, 5)):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel=(5, 5)):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
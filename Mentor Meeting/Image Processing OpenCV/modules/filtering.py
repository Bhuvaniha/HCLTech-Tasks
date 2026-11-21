"""
MODULE 1 - FILTERING (blurring)
===============================

WHAT IS FILTERING?
    Sliding a small window over the image and replacing each pixel with a new
    value calculated from the pixels around it.

WHY?
    Real photos contain noise (grainy dots). Noise confuses edge detection
    and thresholding, so we smooth the picture first.

THE THREE FILTERS HERE (window size is always odd, e.g. 5x5):
    average  = every neighbour counts the same -> softest blur
    gaussian = centre counts more than the edges -> the usual choice
    median   = takes the middle value -> best at removing single dots,
               and it keeps edges sharp
"""


import cv2


def apply_average_blur(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_blur(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(image, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(image, kernel_size)
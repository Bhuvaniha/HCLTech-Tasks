"""
MODULE 4 - THRESHOLDING
=======================

WHAT IS THRESHOLDING?
    Turning a greyscale image into a BLACK & WHITE image by asking every
    pixel one question: "am I brighter or darker than the threshold?"

        brighter -> WHITE (255)  -> the object
        darker   -> BLACK (0)    -> the background

    All the grey detail is thrown away. That is the point: it separates the
    object from the background so later steps have an easy job.

WHY GREYSCALE FIRST?
    The threshold is a single brightness number, and a colour image has three
    channels (blue, green, red) with no one brightness value. So we convert
    first with cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).

THE THREE METHODS
    global   - one fixed number for the whole picture. Simple, but breaks
               when one side is bright and the other side is dark.
    otsu     - the computer picks the number for you by looking at which
               split makes the two groups (dark / bright) differ most.
    adaptive - a different number for every small area, taken from the
               pixels around it. Best for shadows and uneven lighting.
"""


import cv2


def global_threshold(image, threshold_value=127):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(image, threshold_value, 255,
                                    cv2.THRESH_BINARY)
    return binary_image


def otsu_threshold(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adding THRESH_OTSU makes cv2.threshold() pick the value for us and
    # return it, so we pass 0 as the threshold and ignore it.
    otsu_value, binary_image = cv2.threshold(image, 0, 255,
                                              cv2.THRESH_BINARY +
                                              cv2.THRESH_OTSU)

    print(f"    Otsu picked the threshold value: {otsu_value:.0f}")
    return binary_image


def adaptive_threshold(image, block_size=11, constant_c=2):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # block_size must be odd. Bigger C means more white pixels (thicker text).
    return cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 block_size, constant_c)
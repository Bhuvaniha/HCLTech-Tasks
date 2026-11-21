"""
The modules folder. One file per image-processing topic:

    filtering.py      -> blurring (removes noise)
    edge_detection.py -> finding boundaries between objects
    morphology.py     -> cleaning up a black & white image
    thresholding.py   -> turning a picture into black & white

main.py imports the functions it needs from these files:

    from modules.filtering import apply_gaussian_blur
"""
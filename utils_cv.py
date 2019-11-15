import os
import numpy
import cv2 as cv
import numpy as np

def applyLaplace(image, debug = False):
    ## image is the 2D array with 3/4 channels

    ddepth = cv.CV_16S
    kernel_size = 3

    # apply gaussian to reduce the false positives
    image = cv.GaussianBlur(image, (3, 3), 0)

    # Convert the image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Laplace function
    dst = cv.Laplacian(image_gray, ddepth, ksize=kernel_size)

    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)

    # debugging
    if debug == True:
        window_name = "Laplace Demo"
        # Create Window
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        cv.imshow(window_name, abs_dst)
        cv.waitKey(0)
    return abs_dst

def meanOfLaplace(image):
    ## image is the 2D array of 3/4 channels

    dst = applyLaplace(image)
    return np.mean(dst)

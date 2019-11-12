import argparse
import os
import numpy
import cv2
import imutils
from imutils import paths
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def main(filedir):
    # tutoral: https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", type=str, required=True,
    	help="path to input directory of images to stitch")
    ap.add_argument("-o", "--output", type=str, required=True,
    	help="path to the output image")
    args = vars(ap.parse_args())

    # grab the paths to the input images and initialize our images list
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(args["images"])))
    images = []

    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    numImages = 8
    #numImages = len(imagePaths)
    for ii in range(0, numImages):
    	print("image found", ii)
    	image = cv2.imread(imagePaths[ii])
    	images.append(image)

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    print(status)
    print(stitched)
    plt.show()

    return

if __name__ == '__main__':
    main('Images')

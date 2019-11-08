import os
import matplotlib.image as mpimg
import numpy as np
import util


def getRGBRatio(img):
    # img = cv2.imread(filename) # BGR
    # img = util.read_jpeg(filename)
    # Assume img is RGB
    rows,cols, channels = img.shape
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    R2 = np.power(R, 1 * np.ones(R.shape))
    G2 = np.power(G, 1 * np.ones(G.shape))
    B2 = np.power(B, 1 * np.ones(B.shape))


    luminance = 0.299 * R2 + 0.587 * G2 + 0.114 * B2
    luminance = luminance / (3 * 255)

    # R = np.power(R, 2.2 * np.ones(R.shape))
    # G = np.power(G, 2.2 * np.ones(G.shape))
    # B = np.power(B, 2.2 * np.ones(B.shape))

    meanR = np.mean(R[:,:])
    meanG = np.mean(G[:,:])
    meanB = np.mean(B[:,:])
    meanLuminance = np.mean(luminance)

    return meanLuminance, meanG/meanR, meanG/meanB
    # return meanLuminance, meanG

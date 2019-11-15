import numpy as np
from scipy import ndimage
import utils_cv


def extract_features(img):
    # img = cv2.imread(filename) # BGR
    # img = util.read_jpeg(filename)
    # Assume img is RGB
    rows,cols, channels = img.shape
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    R2 = np.power(R, 1 * np.ones(R.shape))
    G2 = np.power(G, 1 * np.ones(G.shape))
    B2 = np.power(B, 1 * np.ones(B.shape))

    luminance = 0.299 * R2 + 0.587 * G2 + 0.114 * B2
    luminance = luminance / ((0.299 + 0.587 + 0.114) * 255)

    # R = np.power(R, 2.2 * np.ones(R.shape))
    # G = np.power(G, 2.2 * np.ones(G.shape))
    # B = np.power(B, 2.2 * np.ones(B.shape))

    meanR = np.mean(R[:,:])
    meanG = np.mean(G[:,:])
    meanB = np.mean(B[:,:])
    meanLuminance = np.mean(luminance[:, :])

    return meanLuminance, meanG/meanR, meanG/meanB, meanB/255, meanR/255, utils_cv.meanOfLaplace(img), 1
    # return meanLuminance, meanG


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def laplacian(img):
    img = ndimage.gaussian_filter(img, 3)

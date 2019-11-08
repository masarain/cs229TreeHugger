import os
import matplotlib.image as mpimg
import numpy as np
import util


def getRGBRatio(filename):
      # img = cv2.imread(filename) # BGR
      img = util.read_jpeg(filename)
      rows,cols, channels = img.shape
      R = img[:,:,0]
      G = img[:,:,1]
      B = img[:,:,2]
      meanR = np.mean(R[:,:])
      meanG = np.mean(G[:,:])
      meanB = np.mean(B[:,:])
      return meanG/meanR, meanG/meanB

import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../../features.csv", header=None)
features = []
for row in df.iterrows():
    features.append((row[1][0], list(row[1][1:])))

features = [feature[1] for feature in sorted(features, key=lambda x: int(x[0][6:]))]

print(kmeans(whiten(np.array(features)), 16))

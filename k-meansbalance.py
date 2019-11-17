import numpy as np

from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("features.csv", header=None)
train_features = []
test_features = []

train_num = 450
test_num = 50

i = 0
for row in df.iterrows():
    # if i < train_num:
    train_features.append((row[1][0], list(row[1][1:])))
    # else: 
    test_features.append((row[1][0], list(row[1][1:])))
    # i += 1

train_features = [feature[1] for feature in sorted(train_features, key=lambda x: int(x[0][6:]))]

test_features = train_features[train_num: train_num + test_num]
train_features = train_features[: train_num]

centroids = (kmeans(whiten(np.array(train_features)), 2))[0]
centroid1, centroid2 = centroids[0], centroids[1] # note that this is hard coded 


centroid_1_label = 0
centroid_2_label = 0

train_labels = []
test_labels = []

labels = pd.read_csv("./train_v2.csv") # labels are whether or not image is any sort of cloudy or haze


for i in range(train_num + test_num):
    tags = labels.iloc[i]["tags"]
    if i < train_num:
        train_labels.append(int("water" not in tags))
    else:
        test_labels.append(int("water" not in tags))

centroid1count = 0
centroid2count = 0
for i, features in enumerate(train_features):
    norm1 = np.linalg.norm(centroid1.reshape(7,1) - np.array(features).reshape(7,1))
    norm2 = np.linalg.norm(centroid2.reshape(7,1) - np.array(features).reshape(7,1))
    if norm1 > norm2:
        centroid_1_label += train_labels[i]
        centroid1count += 1
    else: 
        centroid_2_label += train_labels[i]
        centroid2count += 1



print(centroid_1_label)
print(centroid_2_label)

centroid_1_label = 1
centroid_2_label = 0
num_correct = 0
for i, features in enumerate(test_features):
    norm1 = np.linalg.norm(centroid1.reshape(7,1) - np.array(features).reshape(7,1))
    norm2 = np.linalg.norm(centroid2.reshape(7,1) - np.array(features).reshape(7,1))
    if norm1 > norm2:
        if test_labels[i] == centroid_1_label: num_correct += 1
        else: print("wrong in norm1!: predicted: {}. actual: {}".format(1, test_labels[i]))
    else: 
        if test_labels[i] == centroid_2_label: num_correct += 1
        else: print("wrong in norm2!: predicted: {}. actual: {}".format(0, test_labels[i]))
    
print(num_correct)

import numpy as np

from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("features.csv", header=None)
train_features = []
test_features = []

train_num = 38000
test_num = 2000

    
for row in df.iterrows():
    train_features.append((row[1][0], list(row[1][1:])))
    test_features.append((row[1][0], list(row[1][1:])))

train_features = [feature[1] for feature in sorted(train_features, key=lambda x: int(x[0][6:]))]

test_features = train_features[train_num: train_num + test_num]
train_features = train_features[: train_num]

labels = pd.read_csv("./train_v2.csv") # labels are whether or not image is any sort of cloudy or haze

train_labels = []
test_labels = []

# creating the labels
for i in range(train_num + test_num):
    tags = labels.iloc[i]["tags"]
    if i < train_num:
        train_labels.append(int("cloudy" not in tags and "haze" not in tags))
    else:
        test_labels.append(int("cloudy" not in tags and "haze" not in tags))




def main():
    centroids = (kmeans(whiten(np.array(train_features)), 2))[0]
    centroid1, centroid2 = centroids[0], centroids[1] 


    centroid1count = 0
    centroid2count = 0
    centroid1val = []
    centroid2val = []
    for i, features in enumerate(train_features):
        norm1 = np.linalg.norm(centroid1.reshape(7,1) - np.array(features).reshape(7,1))
        norm2 = np.linalg.norm(centroid2.reshape(7,1) - np.array(features).reshape(7,1))
        if norm1 > norm2: 
            centroid1count += 1
            centroid1val.append(train_labels[i])
        else: 
            centroid2count += 1
            centroid2val.append(train_labels[i])

    # print("Total Labels of Centroid1 Val {}".format(centroid1val))
    # print("Total Labels of Centroid2 Val {}".format(centroid2val))
    print("Number of images with centroid 1: {}".format(centroid1count))
    print("Number of images with centroid2: {}".format(centroid2count))
    print("Centroid 1: {}".format(centroid1))
    print("Centroid 2: {}".format(centroid2))

    centroid_1_label = int(np.rint(sum(centroid1val)/centroid1count)) == 0
    centroid_2_label = int(np.rint(sum(centroid2val)/centroid2count)) == 0
    # centroid_1_label = 1
    # centroid_2_label = 0
    print("Centroid 1 label {}".format(centroid_1_label))
    print("Centroid 2 label {}".format(centroid_2_label))
    assert(centroid_1_label != centroid_2_label)

    # centroid1 = np.array([1.75618451, 7.54931004, 6.36480105, 1.72823512, 1.41218916, 3.03392502, 1. ])
    # cetnroid2 = np.array( [ 3.46567439, 6.24308837, 6.04587955, 3.36558527, 3.13408703, 2.01998688, 1.])

    num_correct = 0
    norm_1_correct, norm_1_incorrect, norm_2_correct, norm_2_incorrect = 0,0,0,0
    for i, features in enumerate(test_features):
        norm1 = np.linalg.norm(centroid1.reshape(7,1) - np.array(features).reshape(7,1))
        norm2 = np.linalg.norm(centroid2.reshape(7,1) - np.array(features).reshape(7,1))
        if norm1 > norm2:
            if test_labels[i] == centroid_2_label: 
                num_correct += 1
                norm_1_correct +=1 
            else: norm_1_incorrect +=1

            # else: print("wrong in norm1!: predicted: {}. actual: {}".format(1, test_labels[i]))
        else: 
            if test_labels[i] == centroid_1_label: 
                num_correct += 1
                norm_2_correct += 1
            else:
                norm_2_incorrect += 1
            # else: print("wrong in norm2!: predicted: {}. actual: {}".format(0, test_labels[i]))
        
    print("Number correct {} out of {}".format(num_correct, test_num))
    print("Number actually not cloudy or hazy {}".format(sum(test_labels)))
    print("Norm1: {}. Correct: {}. Inncorect: {}".format(centroid_1_label, norm_1_correct, norm_1_incorrect))
    print("Norm2: {}. Correct: {}. Inncorect: {}".format(centroid_2_label, norm_2_correct, norm_2_incorrect))

if __name__ == "__main__":
     for _ in range(10):
        print("######## This is iteration : {} ########".format(_))
        main()
        print("\n\n")
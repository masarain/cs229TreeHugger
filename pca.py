import numpy as np

from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("features.csv", header=None)
train_features = []
test_features = []

train_num = 1000
test_num = 1

    
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




pca = PCA(n_components=2)
X = pca.fit_transform(train_features)
print(pca.explained_variance_ratio_)
# X_1 = pca.components_[0]
# X_2 = pca.components_[1]
X_1 = X[:,0]
X_2 = X[:,1]
train_labels = np.array(train_labels)
x_1 = plt.scatter(X_1[train_labels==1], X_1[train_labels==1], marker = 'x', c= "r", linewidth=2)
x_2 = plt.scatter(X_2[train_labels==0], X_2[train_labels==0], marker = 'o', linewidth=2)
plt.xlabel('u1')
plt.ylabel('u2')
plt.legend((x_1, x_2), ("Not Cloudy","Cloudy"))
plt.savefig("pca.png")
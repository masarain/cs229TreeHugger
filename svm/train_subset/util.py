import pandas as pd
from sklearn import svm
filepath = "train_v2.csv"

df = pd.read_csv(filepath)
df = df.drop([i for i in range(500, df.shape[0])])
labels = set()

for index, row in df.iterrows():
    currLabels = row[1]
    for label in currLabels.split():
        labels.add(label) 

category = dict()
for i, label in enumerate(sorted(labels)):
    category[label] = i


print(category)

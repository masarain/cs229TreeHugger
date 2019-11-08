import pandas as pd
filepath = "train_v2.csv"


df = pd.read_csv(filepath)
df = df.drop([i for i in range(500, df.shape[0])])

labels = set()

for row in df: 
    currLabels = row[1]
    for label in currLabels.split():
        labels.add(label) 

print(labels)


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import imageio
import pandas as pd
import glob, os
import numpy as np

fileDir = os.getcwd()
# os.chdir("./train-jpg")

# there are 40480 training examples
# we will allocate 39000 for training
# and the remaining 1480 will be for validation

input_size = 65536 # 256^2
hidden_size = 5
hidden_size_1 = 4
hidden_size_2 = 3
hidden_size_3 = 2
num_classes = 1
learning_rate = 0.001
num_epochs = 5

train_num = 1000
test_num = 148

# train_num = 39000
# test_num = 1480

# %% Load data--for clouds and non-clouds
images = []

for file in glob.glob("*.jpg"):
    images.append(file)
images = sorted(images, key=lambda filename: int(filename[6: -4])) # string splicing so that the images are in order

train_images = []
test_images = []

train_labels = []
test_labels = []
labels = pd.read_csv("./train_v2.csv") # labels are whether or not image is any sort of cloudy or haze

for i in range(train_num + test_num):
    tags = labels.iloc[i]["tags"]
    if i < train_num:
        # train_images.append(imageio.imread(images[i], as_gray=True).flatten())
        train_labels.append(int("cloudy" not in tags and "haze" not in tags))
        # train_labels.append(int("water" not in tags))
        pass
    else:
        test_images.append(imageio.imread(images[i], as_gray=True).flatten())
        test_labels.append(int("cloudy" not in tags and "haze" not in tags))
        # test_labels.append(int("water" not in tags))
        

# %%
"""
Go through the trianing examples and rebalance the training set
"""
numNeg = train_num - sum(train_labels)
newLabels = []

for i in range(train_num):
    tags = labels.iloc[i]["tags"]
    val = int("cloudy" not in tags and "haze" not in tags)
    if val== 0 or (val == 1 and numNeg != 0):
        newLabels.append(val)
        train_images.append(imageio.imread(images[i], as_gray=True).flatten())
        if newLabels[-1] == 1: numNeg -= 1

train_labels = newLabels
# train





# %%
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        
        # parameters
        
        # weights
        # self.h1 = nn.Sigmoid() # input_size, hidden_size
        # self.o = nn.Sigmoid() # hidden_size, num_classes

        self.h1 = nn.Linear(input_size, hidden_size) 
        self.h2 = nn.Linear(hidden_size, hidden_size_1)
        self.h3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h4 = nn.Linear(hidden_size_2, hidden_size_3)
        self.o = nn.Linear(hidden_size_3, num_classes)  
        self.sigmoid = nn.Sigmoid()

        # self.h1 = nn.Sigmoid() 
        # self.h2 = nn.Sigmoid()
        # self.h3 = nn.Sigmoid()
        # self.h4 = nn.Sigmoid()
        # self.o = nn.Sigmoid()  

    def forward(self, x):
        x = self.h1(x)
        x = self.sigmoid(x)
        # print("doing x: {}".format(x.shape))
        x = self.h2(x)
        x = self.sigmoid(x)
        x = self.h3(x)
        x = self.sigmoid(x)
        x = self.h4(x)
        x = self.sigmoid(x)
        x = self.o(x)
        x = self.sigmoid(x)
        return x

# %%

model = Net(input_size, hidden_size, num_classes) # no device configuration here
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load("model.ckpt"))
# model.eval()
# optimizer = TheOptimizerClass(*args, **kwargs)

# checkpoint = torch.load('./model.ckpt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


total_step = len(train_images)
for epoch in range(num_epochs):
    for i, image in enumerate(train_images):  

        image = torch.Tensor(train_images[i]).reshape(1, 65536)
        label = torch.Tensor([int(train_labels[i])])
        # label = label.long()
        # label = label.reshape(1,1)
        # label = label.squeeze()
        
        # Forward pass
        outputs = model(image)
        outputs = outputs.squeeze(0)
        # outputs.reshape(1,)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# %%

with torch.no_grad():
    correct = 0
    total = 0
    for i, image in enumerate(test_images):
        image = torch.Tensor(test_images[i]).reshape(1, 65536)
        label = torch.Tensor([int(test_labels[i])])
        outputs = model(image)
        outputs = outputs.squeeze(0)
        outputs = 1 if torch.sum(outputs) >= 0.5 else 0
        if outputs == torch.sum(label):
            correct += 1
        elif outputs == 0: 
            print("#############")
            print(i,outputs, torch.sum(label))
        # _, predicted = torch.max(outputs.data, 1)
        # correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_images), 100 * correct / len(test_images)))



# %%

torch.save(model.state_dict(), 'modelcloudsbalance.ckpt')

# %%

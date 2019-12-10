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
import matplotlib.pyplot as plt


fileDir = os.getcwd()

# there are 40000 training examples
# we will allocate 3600 for training
# 2000 will be for validation
# and the remaining 2000 will be for test

input_size = 65536 # 256^2
not_gray = 3*1
hidden_size = 2000
hidden_size_1 = 1500
hidden_size_2 = 1000
hidden_size_3 = 500
num_classes = 3
learning_rate = 0.001
num_epochs = 50

train_num = 36000
test_num = 2000
valid_num = 2000

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
        train_images.append(imageio.imread(images[i]).flatten())
        new_label = np.array([0,0,1])
        
        if 'cloudy' in tags or 'haze' in tags:
            new_label = np.array([1, 0, 0])
        elif 'habitation' in tags or 'agriculture' in tags or \
        'cultivation' in tags or 'conventional_mine' in tags or \
        'selective_logging' in tags or 'artisinal_mine' in tags or \
        'slash_burn' in tags:
            new_label = np.array([0, 1, 0])

        train_labels.append(new_label)
        # train_labels.append(int("cloudy" not in tags and "haze" not in tags))
        # train_labels.append(int("water" not in tags))
    else:
        
        test_images.append(imageio.imread(images[i]).flatten())
        new_label = np.array([0,0,1])
        
        if 'cloudy' in tags or 'haze' in tags:
            new_label = np.array([1, 0, 0])
        elif 'habitation' in tags or 'agriculture' in tags or \
        'cultivation' in tags or 'conventional_mine' in tags or \
        'selective_logging' in tags or 'artisinal_mine' in tags or \
        'slash_burn' in tags:
            new_label = np.array([0, 1, 0])

        test_labels.append(new_label)


# %%
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()

        self.h1 = nn.Linear(input_size * not_gray, hidden_size) 
        self.h2 = nn.Linear(hidden_size, hidden_size_1)
        self.h3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h4 = nn.Linear(hidden_size_2, hidden_size_3)
        self.o = nn.Linear(hidden_size_3, num_classes)  

    def forward(self, x):
        x = torch.sigmoid(self.h1(x))
        # print("doing x: {}".format(x.shape))
        x = torch.sigmoid(self.h2(x))
        x = torch.sigmoid(self.h3(x))
        x = torch.sigmoid(self.h4(x))
        x = F.softmax(self.o(x), 1)
        return x

# %%

model = Net(input_size, hidden_size, num_classes) # no device configuration here
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(comp_device)
# model = model.to(device=comp_device)

criterion = nn.CrossEntropyLoss() #need to change to CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

total_step = len(train_images)
cost_train = []
cost_dev = []

accuracy_train = []
accuracy_dev = []
for epoch in range(num_epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    epoch_loss_valid = 0
    epoch_accuracy_valid = 0
    for i, image in enumerate(train_images):  

        # print([list(train_labels[i]).index(1)])
        image = torch.Tensor(train_images[i]).reshape(1, input_size * not_gray)
        label = torch.Tensor([list(train_labels[i]).index(1)])
        label = label.long()
        # print(label)
        
        # label = label.long()
        # label = label.reshape(1,1)
        # label = label.squeeze()
        
        # Forward pass
        outputs = model(image)
        # outputs = outputs.squeeze(0)
        # print(outputs)
        # outputs.reshape(1,)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data
        curr_accuracy = (torch.argmax(outputs) == torch.argmax(label)).type(torch.float)
        epoch_accuracy += curr_accuracy
        

    for i, image in enumerate(test_images):
        image = torch.Tensor(test_images[i]).reshape(1, input_size* not_gray)
        label = torch.Tensor([list(test_labels[i]).index(1)])
        label = label.long()
        outputs = model(image)
        loss = criterion(outputs, label)
        epoch_loss_valid += loss.data
        # outputs = outputs.squeeze(0)
        curr_accuracy = (torch.argmax(outputs) == torch.argmax(label)).type(torch.float)
        epoch_loss_valid += curr_accuracy

        curr_accuracy = (torch.argmax(outputs) == torch.argmax(label)).type(torch.float)
        epoch_accuracy_valid += curr_accuracy

    print ("Epoch [{}/{}], AverageLoss {}, Average Accuracy {}"
                   .format(epoch+1, num_epochs, epoch_loss/len(train_images),epoch_accuracy/len(train_images)))
    cost_train.append(epoch_loss/len(train_images))
    cost_dev.append(epoch_loss_valid/len(test_images))
    accuracy_train.append(epoch_accuracy/len(train_images))
    accuracy_dev.append(epoch_accuracy_valid/len(test_images))



fig, (ax1, ax2) = plt.subplots(2, 1)
t = np.arange(num_epochs)
ax1.plot(t, cost_train,'r', label='train')
ax1.plot(t, cost_dev, 'b', label='dev')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.set_title('CNN')
ax1.legend()

ax2.plot(t, accuracy_train,'r', label='train')
ax2.plot(t, accuracy_dev, 'b', label='dev')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend()
fig.savefig('./' + 'nn' + '.pdf')

# Validating the model 

exit()
with torch.no_grad():
    correct = 0
    total = 0
    validation_accuracy = 0
    for i, image in enumerate(test_images):
        image = torch.Tensor(test_images[i]).reshape(1, input_size* not_gray)
        label = torch.Tensor([(test_labels[i])])
        outputs = model(image)
        # outputs = outputs.squeeze(0)
        curr_accuracy = (torch.argmax(outputs) == torch.argmax(label)).type(torch.float)
        correct += curr_accuracy
        # outputs = 1 if torch.sum(outputs) >= 0.5 else 0
        # if outputs == torch.sum(label):
        #     correct += 1
        # elif outputs == 0: 
            # print("#############")
            # print(i,outputs, torch.sum(label))
        # _, predicted = torch.max(outputs.data, 1)
        # correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_images), 100 * correct / len(test_images)))

torch.save(model.state_dict(), 'modelclouds.ckpt')
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 256, 256)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 12
        self.conv1 = torch.nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #18 * 128 * 128 = 294912 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(12 * 128 * 128, 64)
        
        #64 input features, 2 output features for our 3 defined classes
        self.fc2 = torch.nn.Linear(64, 3)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 256, 256) to (18, 256, 256)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 256, 256) to (18, 128, 128)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 128, 128) to (1, 294912)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 12 * 128 * 128)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 294912) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 3)
        x = self.fc2(x)
        return(x)


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import util
import time

from kaggledataset import KaggleAmazonDataset 

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from simplecnn import SimpleCNN

def createLossAndOptimizer(net, learning_rate=0.001):
    
	#Loss function
	loss = torch.nn.CrossEntropyLoss()

	#Optimizer
	optimizer = optim.Adam(net.parameters(), lr=learning_rate)

	return(loss, optimizer)


def prune_dict(label_dict, images):
	new_dict = {}
	for key in images.keys():
		new_dict[key] = label_dict[key]
	return new_dict


def create_one_hot_label(labels, classes):
	n_classes = len(classes)
	n_labels = len(labels)

	label_tensors = {}
	transform = transforms.Compose([transforms.ToTensor()])
	for key in labels.keys():
		label = np.zeros(2, dtype = np.int64)
		if labels[key] == 'cloudy':
			label[0] = 1
		elif labels[key] == 'not_cloudy':
			label[1] = 1
		label_tensors[key] = torch.from_numpy(label)

	return label_tensors
	

def preprocess(image_path, label_path, count = 40000):
	images, label_dict = util.load_data_and_label(
		image_path, 
		label_path, 
		count)
	labels = prune_dict(label_dict, images)	
	classes = ('cloudy', 'not_cloudy')
	label_tensors = create_one_hot_label(labels, classes)

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dset = KaggleAmazonDataset(images, label_tensors, transform)

	return dset

	
	
def trainNet(net, train_loader, val_loader, batch_size, n_epochs, learning_rate):
    
	#Print all of the hyperparameters of the training iteration:
	print("===== HYPERPARAMETERS =====")
	print("batch_size=", batch_size)
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	print("=" * 30)

	#Get training data
	n_batches = len(train_loader)

	#Create our loss and optimizer functions
	loss, optimizer = createLossAndOptimizer(net, learning_rate)

	#Time for printing
	training_start_time = time.time()
    
	#Loop for n_epochs
	for epoch in range(n_epochs):

		running_loss = 0.0
		print_every = n_batches // 10
		start_time = time.time()
		total_train_loss = 0
        
		for i, data in enumerate(train_loader, 0):
            
			#Get inputs
			inputs, labels = data

			#Wrap them in a Variable object
			#print("variable input", Variable(inputs))
			#print("var labels", Variable(labels))
			inputs, labels = (Variable(inputs), Variable(labels))

			#Set the parameter gradients to zero
			optimizer.zero_grad()

			#Forward pass, backward pass, optimize
			outputs = net(inputs)
			#print("outputs", outputs)
			loss_size = loss(outputs, torch.max(labels, 1)[1])
			loss_size.backward()
			optimizer.step()

			#Print statistics
			print("loss_size", loss_size)
			running_loss += loss_size.data
			total_train_loss += loss_size.data

			#Print every 10th batch of an epoch
			if (i + 1) % (print_every + 1) == 0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
				#Reset running loss and time
				running_loss = 0.0
				start_time = time.time()
            
		#At the end of the epoch, do a pass on the validation set
		total_val_loss = 0
		for inputs, labels in val_loader:

			#Wrap tensors in Variables
			inputs, labels = Variable(inputs), Variable(labels)

			#Forward pass
			val_outputs = net(inputs)
			val_loss_size = loss(val_outputs, torch.max(labels, 1)[1])
			total_val_loss += val_loss_size.data
            
		print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
	print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def main():
	print("hello world")

	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)

	## 1. preprocess
	image_path = '/home/mharwell/Projects/planet/train-jpg'
	label_path = '/home/mharwell/Projects/planet/train_v2_cloudy_not_cloudy.csv'
	n_exp = int(12800 / 2)

	train_set = preprocess(image_path, label_path, n_exp)

	# Training
	n_training_samples = int(.80 * n_exp)
	train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

	#Validation
	n_val_samples = n_exp - n_training_samples#int(.20 * n_exp)
	val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

	#Test
	#n_test_samples = 5000
	#test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


	batch_size = 128
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
						sampler=train_sampler, num_workers=2)

	#test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
	val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 							sampler=val_sampler, num_workers=2)


	CNN = SimpleCNN()
	trainNet(CNN, train_loader, val_loader, batch_size=batch_size, n_epochs=5, learning_rate=0.001)



	return

if __name__ == '__main__':
	main()

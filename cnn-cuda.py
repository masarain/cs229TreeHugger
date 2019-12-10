import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import util
import time
import argparse
import matplotlib.pyplot as plt

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
	optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-3)

	return(loss, optimizer)


def prune_dict(label_dict, images):
	new_dict = {}
	for key in images.keys():
		new_dict[key] = label_dict[key]
	return new_dict


def preprocess(image_path, label_path, count = 40000):

	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dset = KaggleAmazonDataset( image_path, label_path, transform)
	#print("labels len", util.load_labels(label_path))
	return dset


def accuracy(labels, outputs):
    labels = labels.to('cpu')
    outputs = outputs.to('cpu')
    #print("labels", torch.max(labels, 1)[1])
    #print("outputs", torch.max(outputs, 1)[1])
    checks = (torch.max(outputs, 1)[1] == torch.max(labels, 1)[1]).type(torch.float)
    #print("checks", checks)
    correct = (torch.sum(checks))
    return correct.data, len(checks)


def accuracyByClass(labels, outputs, label):
    labels = labels.to('cpu')
    outputs = outputs.to('cpu')
    checks = ((torch.max(outputs, 1)[1] == label)).type(torch.int)
    count = (torch.max(labels, 1)[1] == label).type(torch.int)
    checks = ((checks & count)).type(torch.float)
    
    return (torch.sum(checks).data), torch.sum(count).data


def specificityByClass(labels, outputs, label):
    labels = labels.to('cpu')
    outputs = outputs.to('cpu')
    checks = ((torch.max(outputs, 1)[1] == label)).type(torch.int)
    count = (torch.max(labels, 1)[1] != label).type(torch.int)
    checks = ((checks & count)).type(torch.float)
    
    return (torch.sum(checks).data), torch.sum(count).data

def trainNet(args, net, train_loader, val_loader, test_loader, stitched_loader, batch_size, n_epochs, learning_rate):

	#Print all of the hyperparameters of the training iteration:
	print("===== HYPERPARAMETERS ========")
	print("batch_size=", batch_size)
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	print("device = %s" % ('CUDA' if args.device==torch.device('cuda') else 'CPU'))
	print("=" * 30)

	net = net.to(device=args.device)

	#Get training data
	n_batches = len(train_loader)

	#Create our loss and optimizer functions
	loss, optimizer = createLossAndOptimizer(net, learning_rate)

	#Time for printing
	training_start_time = time.time()
	
	#vars
	cost_train = []
	cost_dev = []
	accuracy_train = []
	accuracy_dev = []

	#Loop for n_epochs
	for epoch in range(n_epochs):

		running_loss = 0.0
		print_every = n_batches // 10
		start_time = time.time()
		total_train_loss = 0
		acc = 0.0
		total_acc = 0.0
		total_count = 0
		for i, data in enumerate(train_loader, 0):

			#Get inputs
			inputs, labels = data
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)

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
			#print("output", outputs, np.shape(outputs))
			accu, count = accuracy(labels, outputs)
			total_acc +=  accu
			total_count += count
			#exit(1)
			#Print every 10th batch of an epoch
			if (i + 1) % (print_every + 1) == 0:
				print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
				#Reset running loss and time
				running_loss = 0.0
				acc = 0.0
				start_time = time.time()
			del inputs
			del labels
			torch.cuda.empty_cache()
		print("Train acc = {:.2f}, ".format(total_acc / total_count))
		cost_train.append(total_train_loss / len(train_loader))
		accuracy_train.append(total_acc / total_count)
		#At the end of the epoch, do a pass on the validation set
		total_val_loss = 0
		total_val_acc = 0.0
		total_acc_count = 0
		for inputs, labels in val_loader:
			inputs = inputs.to(args.device)
			labels = labels.to(args.device)

			#Wrap tensors in Variables
			inputs, labels = Variable(inputs), Variable(labels)

			#Forward pass
			val_outputs = net(inputs)
			val_loss_size = loss(val_outputs, torch.max(labels, 1)[1])
			total_val_loss += val_loss_size.data
			acc, count = accuracy(labels, val_outputs)
			total_val_acc += acc
			total_acc_count += count
			del inputs
			del labels
			torch.cuda.empty_cache()
		print("Validation loss = {:.2f}, ".format(total_val_loss / len(val_loader)))
		print("Validation acc = {:.2f}, ".format(total_val_acc / total_acc_count))
		cost_dev.append(total_val_loss / len(val_loader))
		accuracy_dev.append(total_val_acc / total_acc_count)
	print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

	#At the end, do a pass on the test set
	total_test_loss = 0
	total_test_acc = 0.0
	total_test_count = 0
	total_cloud_accuracy = 0.0
	total_cloud_count = 0.0
	total_human_accuracy = 0.0
	total_human_count = 0.0
	total_non_accuracy = 0.0
	total_non_count = 0.0
	for inputs, labels in test_loader:
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)

		#Wrap tensors in Variables
		inputs, labels = Variable(inputs), Variable(labels)

		#Forward pass
		test_outputs = net(inputs)
		test_loss_size = loss(test_outputs, torch.max(labels, 1)[1])
		total_test_loss += test_loss_size.data
		acc, count = accuracy(labels, test_outputs)
		total_test_acc += acc
		total_test_count += count
		acc, count = accuracyByClass(labels, test_outputs, 0)
		total_cloud_accuracy += acc
		total_cloud_count += count
		acc, count = accuracyByClass(labels, test_outputs, 1)
		total_human_accuracy += acc
		total_human_count += count
		acc, count = accuracyByClass(labels, test_outputs, 2)
		total_non_accuracy += acc
		total_non_count += count
		
		del inputs
		del labels
		torch.cuda.empty_cache()
	print("Test loss = {:.2f}, ".format(total_test_loss / len(test_loader)))
	print("Test acc = {:.2f}, ".format(total_test_acc / total_test_count))
	print("Test Cloud acc = {:.2f}, ".format(total_cloud_accuracy / total_cloud_count))
	print("Test Human acc = {:.2f}, ".format(total_human_accuracy / total_human_count))
	print("Test Non-Human acc = {:.2f}, ".format(total_non_accuracy / total_non_count))


	#At the end, do a pass on the stitched set
	total_test_loss = 0
	total_test_acc = 0.0
	total_test_count = 0
	total_cloud_accuracy = 0.0
	total_cloud_count = 0.0
	total_human_accuracy = 0.0
	total_human_count = 0.0
	total_non_accuracy = 0.0
	total_non_count = 0.0
	for inputs, labels in stitched_loader:
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)

		#Wrap tensors in Variables
		inputs, labels = Variable(inputs), Variable(labels)

		#Forward pass
		test_outputs = net(inputs)
		test_loss_size = loss(test_outputs, torch.max(labels, 1)[1])
		total_test_loss += test_loss_size.data
		acc, count = accuracy(labels, test_outputs)
		total_test_acc += acc
		total_test_count += count
		acc, count = accuracyByClass(labels, test_outputs, 0)
		total_cloud_accuracy += acc
		total_cloud_count += count
		acc, count = accuracyByClass(labels, test_outputs, 1)
		total_human_accuracy += acc
		total_human_count += count
		acc, count = accuracyByClass(labels, test_outputs, 2)
		total_non_accuracy += acc
		total_non_count += count
		
		del inputs
		del labels
		torch.cuda.empty_cache()
	print("Stitched loss = {:.2f}, ".format(total_test_loss / len(stitched_loader)))
	print("Stitched acc = {:.2f}, ".format(total_test_acc / total_test_count))
	print("Stitched Cloud acc = {:.2f}, ".format(total_cloud_accuracy / total_cloud_count))
	print("Stitched Human acc = {:.2f}, ".format(total_human_accuracy / total_human_count))
	print("Stitched Non-Human acc = {:.2f}, ".format(total_non_accuracy / total_non_count))

	## plotting the loss and accuracy:
	fig, (ax1, ax2) = plt.subplots(2, 1)
	t = np.arange(n_epochs)
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
	
	fig.savefig('./' + 'cnn-cuda_reduced_neurons' + '.pdf')
	
def main():
	print("hello world")

	parser = argparse.ArgumentParser(description='CNN Training')
	parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
	args = parser.parse_args()
	args.device = None
	if args.enable_cuda and torch.cuda.is_available():
		args.device = torch.device('cuda')
	else:
		args.device = torch.device('cpu')
	torch.cuda.empty_cache()
	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)

	## 1. preprocess
	image_path = r'/home/anishag/tree/cs229/train-jpg'
	label_path = r'/home/anishag/tree/cs229/cs229TreeHugger/train_v2.csv'
	n_exp = int(40000)

	train_set = preprocess(image_path, label_path)

	#Validation
	n_val_samples = 2000
	val_sampler = SubsetRandomSampler(np.arange(0, n_val_samples, dtype=np.int64))

    # Training
	n_training_samples = 36000
	train_sampler = SubsetRandomSampler(np.arange(n_val_samples, n_training_samples + n_val_samples, dtype=np.int64))

	#Test
	n_test_samples = 2000
	test_sampler = SubsetRandomSampler(np.arange(n_training_samples + n_val_samples, n_training_samples + n_val_samples + n_test_samples, dtype=np.int64))
	
	#Test_stitched
	stitched_im_path = r'/home/anishag/tree/cs229/cs229TreeHugger/stitched'
	stitched_csv_path = r'/home/anishag/tree/cs229/cs229TreeHugger/stitched/stitched_labels.csv'
	stitched_set = preprocess(stitched_im_path, stitched_csv_path)
	n_stitched_samples = 2000
	stitched_sampler = SubsetRandomSampler(np.arange(0, n_stitched_samples, dtype=np.int64))


	batch_size = 128
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
						sampler=train_sampler, num_workers=2)

	test_loader = torch.utils.data.DataLoader(train_set, batch_size=16, sampler=test_sampler, num_workers=2)
	val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler, num_workers=2)
	
	stitched_loader = torch.utils.data.DataLoader(stitched_set, batch_size=16, sampler=stitched_sampler, num_workers=2)
	#print("stitched_loader", stitched_loader)
	CNN = SimpleCNN()
	trainNet(args, CNN, train_loader, val_loader, test_loader, stitched_loader, batch_size=batch_size, n_epochs=15, learning_rate=0.001)
	del CNN
	torch.cuda.empty_cache()
	return

if __name__ == '__main__':
	main()

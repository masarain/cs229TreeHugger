import torch
from torch.utils import data

import matplotlib.image as mpimg
import os
import numpy as np

class KaggleAmazonDataset(data.Dataset):
	"""Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space 	competition.

	Arguments:
	A CSV file path
	Path to image folder
	Extension of images
	PIL transforms
	"""

	def __init__(self, image_path, label_path, transform=None):
		self.filenames = os.listdir(image_path)
		self.image_path = image_path
		self.labels = self.load_labels(label_path)
		self.transform = transform
		self.filenames = [k for k in self.filenames if '.jpg' in k]

	def load_labels(self, csv_path):
	    with open(csv_path, 'r') as csv_file:
	        headers = csv_file.readline().strip().split(',')

	    result = {}
	    with open(csv_path, 'r') as csv_file:
	        # Read and throw away the first line.
	        line = csv_file.readline().strip().split(',')
	        line = csv_file.readline().strip().split(',')
	        while line and len(line) == 2:
	            result[line[0]] = line[1]
	            line = csv_file.readline().strip().split(',')

	    return result

	def read_jpeg(self, jpeg_path):
		return mpimg.imread(jpeg_path)

	def __getitem__(self, index):
		filename = self.filenames[index]
		img = self.read_jpeg( self.image_path + '/' + filename)[:,:,0:3]
		if self.transform is not None:
			img = self.transform(img)

		new_label = np.zeros(3, dtype = np.int64)
		label = self.labels[filename[0:-4]] 
		if 'cloudy' in label or 'haze' in label:
			new_label = np.array([1, 0, 0])
		elif 'habitation' in label or 'agriculture' in label or 'cultivation' in label or 'conventional_mine' in label or 'selective_logging' in label or 'artisinal_mine' in label or 'slash_burn' in label:
			new_label = np.array([0, 1, 0])
		else:
		    new_label = np.array([0, 0, 1])

		return img, torch.from_numpy(new_label)

	def __len__(self):
		return len(self.filenames)

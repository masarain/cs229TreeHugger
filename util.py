import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os


def softmax(x):
    results = []
    for i in range(x.shape[0]):
        row = x[i]
        row = row - np.max(row)
        normalization = np.sum(np.exp(row))
        results += [np.exp(row) / normalization]
    return np.array(results)


def load_data_and_label(train_path, label_path, count=40000):
    label_dict = load_labels(label_path)
    images = read_all_jpegs(train_path, count)

    return images, label_dict


def read_all_jpegs(jpeg_folder_path, count=40000):
    filenames = os.listdir(jpeg_folder_path)
    results = {}
    for file in filenames[0:count]:
        if file[len(file) - 4:] == '.jpg':
            results[file[0:-4]] = read_jpeg(jpeg_folder_path + "/" + file)

    return results


def extract_all_feature(jpeg_folder_path, extract_feature_func):
    filenames = os.listdir(jpeg_folder_path)
    results = {}
    i = 0
    for file in filenames:
        if i % 1000 == 0:
            print("Extracting feature on image number: " + str(i))
        i += 1
        if file[len(file) - 4:] == '.jpg':
            img = read_jpeg(jpeg_folder_path + "/" + file)
            results[file[0:-4]] = extract_feature_func(img)

    return results

def read_jpeg(jpeg_path):
    return mpimg.imread(jpeg_path)


def load_labels(csv_path):
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


def get_all_jpeg_files(jpeg_folder_location):
    filenames = os.listdir(jpeg_folder_location)
    results = []
    for file in filenames:
        if file[len(file) - 4:] == '.jpg':
            results += [jpeg_folder_location + "/" + file]

    return results



def plot_points(x, y):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y == 0, :]
    x_two = x[y == 1, :]
    x_three = x[y == -1, :]

    plt.scatter(x_one[:,0], x_one[:,1], marker='x', color='red')
    plt.scatter(x_two[:,0], x_two[:,1], marker='o', color='blue')
    plt.scatter(x_three[:, 0], x_three[:, 1], marker='x', color='green')

    plt.show()


def plot_3d(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_one = x[y == 0, :]
    x_two = x[y == 1, :]

    ax.scatter(x_one[:, 0], x_one[:, 1], x_one[:, 2], c='red')
    ax.scatter(x_two[:, 0], x_two[:, 1], x_two[:, 2], c='blue')

    plt.show()


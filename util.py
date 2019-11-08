import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


def load_data_and_label(train_path, label_path):
    label_dict = load_labels(label_path)
    images = read_all_jpegs(train_path)

    return images, label_dict


def read_all_jpegs(jpeg_folder_path):
    filenames = os.listdir(jpeg_folder_path)
    results = {}
    for file in filenames:
        if file[len(file) - 4:] == '.jpg':
            results[file[0:-4]] = read_jpeg(jpeg_folder_path + "/" + file)

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




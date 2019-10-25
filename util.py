import matplotlib.image as mpimg
import matplotlib.pyplot as plt


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



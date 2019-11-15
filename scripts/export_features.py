import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import getRGBratios
import csv


img_folder = "train_subset"
csv_file = "train_v2.csv"
output_csv_file = "features.csv"


def main(train_path, label_path):
    images, label_dict = util.load_data_and_label(train_path, label_path)

    features_to_csv = []

    for img_name in images.keys():
        features = [img_name]
        features += getRGBratios.getRGBRatio(images[img_name])

        features_to_csv += [features]

    with open(output_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(features_to_csv)


main(img_folder, csv_file)


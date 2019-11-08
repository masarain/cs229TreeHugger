import numpy as np
import math
import util


def main(train_path, label_path):
    images, label_dict = util.load_data_and_label(train_path, label_path)
    print(images["train_1"])
    print(label_dict["train_1"])


main("train_subset", "train_v2.csv")


import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import featureExtract
import logReg


img_folder = "train_subset"
csv_file = "train_v2.csv"
valid_path = "validation_subset"


def extract_feature_and_label(imgs, labels, label_to_include):
    x = []
    y = []
    for img_name in imgs.keys():
        features = featureExtract.extract_features(imgs[img_name])
        if label_to_include in labels[img_name]:
            y += [1]
        else:
            y += [0]
        x += [features]

    return x, y


def main(train_path, label_path):
    images, label_dict = util.load_data_and_label(train_path, label_path)
    valid_imgs, label_dict = util.load_data_and_label(valid_path, label_path)

    labels = ['water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground',
              'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
    accuracy = {}
    positive_count = {}
    true_positive_count = {}
    true_negative_count = {}
    false_negative_count = {}
    false_positive_count = {}

    for label in labels:
        x, y = extract_feature_and_label(images, label_dict, label)

        theta = None
        theta = logReg.logistic_regression(np.array(x), y, theta)

        valid_x, valid_y = extract_feature_and_label(valid_imgs, label_dict, label)
        predicted = logReg.predict(valid_x, theta)

        correct = 0
        correct_non_label = 0
        correct_cloud = 0
        false_positive = 0
        false_negative = 0
        for i in range(len(predicted)):
            if predicted[i] == valid_y[i]:
                correct += 1
                if predicted[i] == 1:
                    correct_cloud += 1
                else:
                    correct_non_label += 1
            elif predicted[i] == 0:
                false_negative += 1
            else:
                false_positive += 1

        general_cloud = np.sum(valid_y)

        accuracy[label] = correct / len(predicted)
        positive_count[label] = general_cloud
        true_positive_count[label] = correct_cloud
        true_negative_count[label] = correct_non_label
        false_positive_count[label] = false_positive
        false_negative_count[label] = false_negative

        print("Accuracy is: " + str(correct / len(predicted)))
        print("General " + label + " count: " + str(general_cloud))
        print("Non " + label + " count: " + str(len(predicted) - general_cloud))
        print("Accurate " + label + ": " + str(correct_cloud))
        print("Accurate non " + label + ": " + str(correct_non_label))
        print("False negative for " + label + ": " + str(false_negative))
        print("False positive for " + label + ": " + str(false_positive))

        # print(predicted)

    for key in accuracy.keys():
        print("Accuracy for " + key + ": " + str(accuracy[key]))
        print("General " + key + " count: " + str(positive_count[key]))
        print("Accurate " + key + ": " + str(true_positive_count[key]))
        print("Accurate non " + key + ": " + str(true_negative_count[key]))
        print("False negative for " + key + ": " + str(false_negative_count[key]))
        print("False positive for " + key + ": " + str(false_positive_count[key]))


main(img_folder, csv_file)


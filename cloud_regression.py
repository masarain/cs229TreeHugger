import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import featureExtract
import logReg

import os
os.chdir("./svm/")


img_folder = "train_subset"
csv_file = "train_v2.csv"
valid_path = "validation_subset"


def extract_feature_and_label(imgs, labels):
    x = []
    y = []
    for img_name in imgs.keys():
        features = featureExtract.extract_features(imgs[img_name])
        if 'hazy' in labels[img_name]:
            continue
        elif 'cloudy' in labels[img_name]:
            y += [1]
        elif 'haze' in labels[img_name]:
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
    labels = ['cloudy']
    accuracy = {}
    positive_count = {}
    true_positive_count = {}

    for label in labels:
        x, y = extract_feature_and_label(images, label_dict)

        # theta = np.array([924.66597612, -96746.61570586, -76539.70693396, 5369.2771662, -2305.59160811, -81622.4210792])
        # theta = np.array([311.67324592, -37.51466762, 11.17806561, 111.50054584, -314.71077383, -14.23669017])
        theta = np.array([829.93898882, -58.36516723, 165.86166034, 1851.25020857, -1401.28024088, 15.04154101, -536.67415106])
        theta = np.array([851.33147301, -72.41160088, 152.9213526, 1825.67382211, -1439.08611494, 3.77706817, -549.94950838])
        theta = np.array([849.57621794, -84.11246563, 136.41558837, 1818.49671105, -1445.57183804, -79.22664165, -564.70006635])
        theta = None
        theta = logReg.logistic_regression(np.array(x), y, theta)

        valid_x, valid_y = extract_feature_and_label(valid_imgs, label_dict)
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


main(img_folder, csv_file)


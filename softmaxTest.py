import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import regression.softmaxReg
import featureExtract
import logReg


img_folder = "train_subset"
csv_file = "train_v2.csv"
valid_path = "validation_subset"


def is_human_intervention(label):
    human_labels = ['habitation', 'agriculture', 'cultivation', 'conventional_mine',
    'selective_logging', 'artisinal_mine', 'slash_burn']

    for human_label in human_labels:
        if human_label in label:
            return True

    return False


def extract_feature_and_label(imgs, labels):
    x = []
    y = []
    for img_name in imgs.keys():
        features = featureExtract.extract_features(imgs[img_name])
        if 'hazy' in labels[img_name]:
            continue
        elif 'cloudy' in labels[img_name] or 'haze' in labels[img_name]:
            y += [[1, 0, 0]]
        elif is_human_intervention(labels[img_name]):
            y += [[0, 1, 0]]
        else:
            y += [[0, 0, 1]]
        x += [features]

    y = np.array(y)
    print("Y shape is: " + str(y.shape))

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

        w = [
        [ 2.17270113, 0.35238994,-2.52509106],
        [ 0.51949841,-2.35026292, 1.83076451],
        [-1.99743922, 1.11609345, 0.88134577],
        [ 2.70263539,-0.58047509,-2.1221603 ],
        [ 1.59282174, 0.91946559,-2.51228733],
        [-0.02840839, 0.10883493,-0.08042655],
        [-0.27839323, 0.42168854,-0.14329531]]


        w = regression.softmaxReg.train(np.array(x), y, w)

        print("w is:")
        print(w)

        valid_x, valid_y = extract_feature_and_label(valid_imgs, label_dict)
        print(valid_y.shape)
        valid_y = np.argmax(valid_y, axis=1)
        predicted = regression.softmaxReg.predict(valid_x, w)

        # print("Predicted:")
        # print(predicted)

        correct = 0
        correct_non_label = 0
        correct_cloud = 0
        false_positive = 0
        false_negative = 0
        for i in range(len(predicted)):
            if predicted[i] == valid_y[i]:
                correct += 1
                if predicted[i] == 0:
                    correct_cloud += 1
                else:
                    correct_non_label += 1
            elif not (predicted[i] == 0):
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


main(img_folder, csv_file)
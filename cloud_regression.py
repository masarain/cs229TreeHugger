import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import getRGBratios


img_folder = "train_subset"
csv_file = "train_v2.csv"


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


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y, theta=None):
    """Train a logistic regression model."""
    if theta is None:
        theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    # theta = np.array([-3000, 0,0])
    max_iterations = 200000

    i = 0
    while i < max_iterations:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            print('Norm is: ' + str(np.linalg.norm(prev_theta - theta)))
            print("Theta is: " + str(theta))
            break
        else:
            if i % 10000 == 0:
                print("Current norm is: " + str(np.linalg.norm(prev_theta - theta)))
                print("Current theta is: " + str(theta))
    return theta


def main(train_path, label_path):
    images, label_dict = util.load_data_and_label(train_path, label_path)
    # print(images["train_1"])
    # print(label_dict["train_1"])

    x = []
    y = []
    general_cloud = 0
    for img_name in images.keys():
        if "partly_cloudy" in label_dict[img_name]:
            continue
        # if "haze" in label_dict[img_name]:
        #     continue
        temp = getRGBratios.getRGBRatio(images[img_name])
        if "haze" in label_dict[img_name] or "cloudy" in label_dict[img_name]:
            y += [1]
            general_cloud += 1
            # Looking for cloud pictures with low luminance values
            # if temp[0] < 0.1:
            #     print(img_name + " with luminance value: " + str(temp[0]))
        # elif "primary" in label_dict[img_name]:
        #     y += [-1]
        else:
            y += [0]

        x += [temp]

    # print(len(y))
    print(np.array(x).shape)
    # print(y)

    # x = np.log(x)

    # plot_points(np.array(x), np.array(y))

    # theta = np.array([-8.19750497e+05, -1.56189738e+07, -1.59023226e+07, -5.36430159e+08, -5.63876556e+08])
    theta = np.array([-98.12421915, -92986.31389205, -84482.43533785, 5291.16535734, -2216.77670579, -75947.0722486])
    theta = np.array([412.76862094, -94944.84688406, -80571.91005116, 5344.90854079, -2249.83723926, -78835.96236183])
    theta = np.array([924.66597612, -96746.61570586, -76539.70693396, 5369.2771662, -2305.59160811, -81622.4210792])
    # theta = logistic_regression(np.array(x), y, theta)
    predicted = -1 * np.matmul(theta, np.transpose(x))
    predicted = np.exp(predicted)
    predicted = 1 / (1 + predicted)

    def round(x):
        if x >= 0.5:
            return 1
        else:
            return 0

    predicted = [round(x) for x in predicted]

    correct = 0
    correct_cloud = 0
    for i in range(len(predicted)):
        if predicted[i] == y[i]:
            correct += 1
            if predicted[i] == 1:
                correct_cloud += 1

    print("Accuracy is: " + str(correct / len(predicted)))
    print("General cloud count: " + str(general_cloud))
    print("Accurate clouds: " + str(correct_cloud))

    # print(predicted)


main(img_folder, csv_file)


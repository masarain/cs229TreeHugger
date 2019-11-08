import numpy as np
import math
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import getRGBratios


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

    i = 0
    while True:
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
    for img_name in images.keys():
        if "partly_cloudy" in label_dict[img_name]:
            continue
        # if "haze" in label_dict[img_name]:
        #     continue
        temp = getRGBratios.getRGBRatio(images[img_name])
        if "haze" in label_dict[img_name] or "cloudy" in label_dict[img_name]:
            y += [1]
            if temp[0] < 0.1:
                print(img_name + " with luminance value: " + str(temp[0]))
        elif "primary" in label_dict[img_name]:
            y += [-1]
        else:
            y += [0]


        x += [temp]

    print(len(y))
    # print(np.array(x))
    # print(y)

    # x = np.log(x)

    plot_points(np.array(x), np.array(y))
    # logistic_regression(np.array(x), y)


main("train_subset", "train_v2.csv")


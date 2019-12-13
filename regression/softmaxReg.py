import numpy as np


def get_loss(w, x, y):
    num_samples = x.shape[0]
    scores = np.dot(x, w)
    prob = softmax(scores)
    loss = (-1 / num_samples) * np.sum(y * np.log(prob)) # + (1 / 2) * np.sum(w*w)
    grad = (-1 / num_samples) * np.dot(x.T, (y - prob)) # + w
    return loss,grad


def softmax(x):
    results = []
    for i in range(x.shape[0]):
        row = x[i]
        row = row - np.max(row)
        normalization = np.sum(np.exp(row))
        results += [np.exp(row) / normalization]
    return np.array(results)


def train(data, labels, w=None, valid_data=None, valid_labels=None):
    if w is None:
        w = np.zeros([data.shape[1], labels.shape[1]])
    converged = False
    iteration_max = 100000
    learning_rate = 1e-1
    i = 0
    all_loss = []
    all_train_accuracy = []
    all_valid_accuracy = []
    while not converged and (i < iteration_max):
        loss, grad = get_loss(w, data, labels)
        w = w - (learning_rate  * grad)

        if i % 200 == 0:
            train_prediction = predict(data, w)
            train_accuracy = calculate_accuracy(train_prediction, np.argmax(labels, axis=1))
            all_train_accuracy += [train_accuracy]
            print("Train accuracy is: " + str(train_accuracy))
            if not valid_labels is None:
                # valid_prediction = predict(valid_labels, w)
                valid_accuracy = calculate_accuracy(predict(valid_data, w), np.argmax(valid_labels, axis=1))
                all_valid_accuracy += [valid_accuracy]
                print("Valid accuracy is: " + str(valid_accuracy))

            all_loss += [loss]
            print("Loss on iteration " + str(i) + " is: " + str(loss))

        if np.sum(grad * grad) < 1e-5:
            converged = True

        i += 1

    return w, all_loss, all_train_accuracy, all_valid_accuracy


def predict(x, w):
    probabilities = softmax(np.dot(x, w))
    prediction = np.argmax(probabilities, axis=1)
    return prediction


def calculate_accuracy(prediction, actual):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            correct += 1
    return correct / len(prediction)




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


def train(data, labels, w):
    # w = np.zeros([data.shape[1], labels.shape[1]])
    converged = False
    iteration_max = 100000
    learning_rate = 1e-2
    i = 0
    while not converged and (i < iteration_max):
        loss, grad = get_loss(w, data, labels)
        w = w - (learning_rate  * grad)

        if i % 1000 == 0:
            print("Loss on iteration " + str(i) + " is: " + str(loss))

        if np.sum(grad * grad) < 1e-5:
            converged = True

        i += 1

    return w


def predict(x, w):
    probabilities = softmax(np.dot(x, w))
    print(probabilities)
    prediction = np.argmax(probabilities, axis=1)
    return prediction



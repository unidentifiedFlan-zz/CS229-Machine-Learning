import numpy as np
import matplotlib.pyplot as plt


def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x):
    """
    Compute softmax function for input.
    Use tricks from previous assignment to avoid overflow
    """
    c = np.max(x, axis =1, keepdims=True)
    top = np.exp(x-c)
    bottom = np.sum(np.exp(x-c), axis=1, keepdims=True)
    s=top/bottom
    return s


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    s=1/(1+np.exp(-x))
    return s


def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    z1 = data.dot(W1) + b1
#    print("z1 shape" + str(z1.shape))
    h = sigmoid(z1)
#    print("h shape" + str(h.shape))
    z2 = h.dot(W2) + b2
    y = softmax(z2)
#    print("y shape" + str(y.shape))
    m = data.shape[0]

#    print("labels shape" + str(labels.shape))
    cost = -(1/m)*np.sum(np.multiply(labels, np.log(y)))

    return h, y, cost


def backward_prop(data, labels, h, params, lam):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    sum_y = 1#np.sum(labels, axis=1) # =1 for one-hot
    z2 = h.dot(W2) + b2
    delta2 = softmax(z2)*sum_y - labels   #D*K
#    print("delta2 shape" + str(delta2.shape))
    gradW2 = np.transpose(h).dot(delta2) #num_hidden*K
    gradb2 = np.sum(delta2, axis=0)

#    print("gradw2 shape" + str(gradW2.shape))
    delta1 = np.multiply(h*(np.ones(h.shape)-h), np.dot(delta2, W2.T)) #To get D*num_hidden matrix
#    print("delta1 shape" + str(delta1.shape))
    gradW1 = data.T.dot(delta1) #num_features*num_hidden
    gradb1 = np.sum(delta1, axis=0)

    B = data.shape[0]
    grad = {}
    grad['W1'] = (gradW1 + 2*lam*W1)/B
    grad['W2'] = (gradW2 + +2*lam*W2)/B
    grad['b1'] = gradb1/B
    grad['b2'] = gradb2/B

    return grad


def update_weights(grad, params, learning_rate):
    params['W1'] -= learning_rate*grad['W1']
    params['W2'] -= learning_rate*grad['W2']
    params['b1'] -= learning_rate*grad['b1']
    params['b2'] -= learning_rate*grad['b2']


def initialise_params(num_hidden, num_features, num_outputs):
    params = {}
    mu, sigma = 0, 0.1
    params['W1'] = np.random.normal(mu, sigma, (num_features, num_hidden))
    params['W2'] = np.random.normal(mu, sigma, (num_hidden, num_outputs))
    params['b1'] = np.zeros(num_hidden)
    params['b2'] = np.zeros(num_outputs)

    return params


def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_samples = trainData.shape[0]
    num_features = trainData.shape[1]
    num_outputs = 10
    num_hidden = 300
    learning_rate = 5
    lam = 0.0001
    params = initialise_params(num_hidden, num_features, num_outputs)

    B = 1000 #batch size
    total_iterations = int(num_samples/B)
    total_epochs = 30

    train_acc_arr = []
    epochs = []
    dev_acc_arr = []
    for j in range(total_epochs):
        for i in range(total_iterations):
            iterData = trainData[i*B:(i*B)+B, :]
            iterLabels = trainLabels[i*B:(i*B)+B, :]
            h, y, cost = forward_prop(iterData, iterLabels, params)
            grad = backward_prop(iterData, iterLabels, h, params, lam)
            update_weights(grad, params, learning_rate)

        epochs.append(j)
        train_accuracy = nn_test(trainData, trainLabels, params)
        train_acc_arr.append(train_accuracy)
        print('Train accuracy: %f' % train_accuracy)
        dev_accuracy = nn_test(devData, devLabels, params)
        dev_acc_arr.append(dev_accuracy)
        print('Validation accuracy: %f' % dev_accuracy)

    plt.figure()
    plt.plot(epochs, train_acc_arr, color='r')
    plt.plot(epochs, dev_acc_arr, color='b')
    plt.show()
    return params


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)

    data_path = "C:/Users/scday/Documents/coding/Machine_Learning/CS229/ps4/DeepLearning/"

    trainData, trainLabels = readData(data_path + 'images_train.csv', data_path + 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    devData = trainData[0:10000, :]
    devLabels = trainLabels[0:10000, :]
    trainData = trainData[10000:, :]
    trainLabels = trainLabels[10000:, :]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData(data_path + 'images_test.csv', data_path + 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    params = nn_train(trainData, trainLabels, devData, devLabels)

    readyForTesting = False
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print('Test accuracy: %f' % accuracy)


if __name__ == '__main__':
    main()
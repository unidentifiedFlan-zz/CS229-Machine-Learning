from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        norm = np.linalg.norm(prev_theta - theta)

        if i % 100000 == 0:
            print('Finished {0} iterations; Diff theta: {1}; theta: {2}; Grad: {3}'.format(
                i, norm, theta, grad))
        if norm < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return theta

def main():
    dir = "Machine_Learning/CS229/ps2/"
    print('==== Training model on data set A ====')
    Xa, Ya = load_data(dir+'data_a.txt')
    thetaA = logistic_regression(Xa, Ya)

    a_df = pd.read_csv(dir + 'data_a.txt', header=None, sep=' ', names=['label', 'x1', 'x2'])
    #fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

 #   for k, theta in enumerate(thetaA):
    ax = plt.axes()
    a_df.query('label == 1').plot.scatter(x='x1', y='x2', ax=ax, color='red')
    a_df.query('label == -1').plot.scatter(x='x1', y='x2', ax=ax, color='blue')
    x = np.arange(0,1,0.1)
    y = -(thetaA[0] + thetaA[1]*x)/thetaA[2]
    ax.plot(x, y)
    plt.show()

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data(dir+'data_b.txt')
    thetaB = logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()

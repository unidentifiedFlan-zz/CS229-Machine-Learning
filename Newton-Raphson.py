import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

data_dir = "C:/Users/scday/Documents/coding/Machine_Learning/CS229/ps1/"
train_input_data_file = "logistic_x.txt"
train_value_data_file = "logistic_y.txt"

X = pd.read_csv(data_dir + train_input_data_file, sep='\ +', header=None, engine='python')
y = pd.read_csv(data_dir + train_value_data_file, sep='\ +', header=None, engine='python').values
y = y.astype(int)

df_X = X
X = X.values

X = np.hstack([np.ones((X.shape[0], 1)), X])
theta = np.zeros((X.shape[1], 1))
theta_seq = []
iters = 10
for i in range(1, iters):
    z = np.multiply(y, X.dot(theta))
    expTerm = 1/(1 + np.exp(-z))
    gradJ = np.mean(np.multiply(np.multiply(y,X), (expTerm - 1)), axis=0)
    gradJ = np.asmatrix(gradJ).T
    H = np.zeros((theta.shape[0], theta.shape[0]))
    for j in range(H.shape[0]):
        for k in range(H.shape[0]):
            H[j][k] = np.mean(np.multiply(np.multiply(X[:, j], X[:, k]), np.multiply(expTerm, (1-expTerm))))

    theta = theta - np.linalg.inv(H).dot(gradJ)
    theta_seq.append(theta)

print(theta)

df_X['label'] = y[:, 0]
ax = plot.axes()
df_X.query('label == -1').plot.scatter(x=0, y=1, ax=ax, color='blue')
df_X.query('label == 1').plot.scatter(x=0, y=1, ax=ax, color='red')

_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
for k, theta in enumerate(theta_seq):
    _ys = (np.asarray(theta)[0] + np.asarray(theta)[1] * _xs) / (- np.asarray(theta)[2])
    plot.plot(_xs, _ys, label='iter {0}'.format(k + 1), lw=0.5)
plot.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plot.show()
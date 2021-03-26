import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


right_threshold = 1300
right_index = 150
num_neighbours = 3


def construct_w(X, x_val):
    tau = 5
    W = np.diag(np.exp(-(X[1, :] - x_val)**2/(tau**2)))

    return W


def smooth_values(X, Y):
    SmoothedY = []
    for i in range(Y.shape[0]):
        y = Y[i, :]
        SmoothedYi = []
        for x_val in X[1]:
            W = construct_w(X, x_val)
            WeightedTheta = np.linalg.inv(X.dot(W.dot(X.T))).dot(X.dot(W.dot(y.T)))
            SmoothedYi.append(WeightedTheta[0] + WeightedTheta[1] * x_val)

        SmoothedY.append(SmoothedYi)
    return SmoothedY


def spectra_distance(f1, f2):
    distance = sum((f1 - f2)**2)
    return distance


def ker(t):
    return max(1-t, 0)


def max_distance_spectra(selected_spectra, spectra):
    max_distance = 0
    for f in spectra:
        d = spectra_distance(f, selected_spectra)
        if d > max_distance:
            max_distance = d
    return max_distance


def spectra_estimate(spectra_neighbours, spectra_right):
    spectra_n_left = spectra_neighbours[:, :right_index-1]
    spectra_n_right = spectra_neighbours[:, right_index::]
    normalisation = 0
    unnormalised_spectra_left_estimate = np.zeros(spectra_n_left.shape)

    for i in range(spectra_n_right.shape[0]):
        ker_val = ker(spectra_distance(spectra_n_right[i], spectra_right)/max_distance_spectra(spectra_n_right[i], spectra_right))
        normalisation += ker_val
        unnormalised_spectra_left_estimate += ker_val*spectra_n_left[i]

    spectra_left_estimate = unnormalised_spectra_left_estimate/normalisation
    return np.asarray(spectra_left_estimate)


def find_spectra_neighbours(spectra_set, spectra):
    spectra_set_right = spectra_set[:, right_index::]
    distances = []
    for i in range(spectra_set_right.shape[0]):
        d = spectra_distance(spectra_set_right[i], spectra)
        distances.append([d, i])

    distances.sort()
    neighbours = []
    for i in range(num_neighbours):
        neighbours.append(spectra_set[distances[i][1]])

    return neighbours

data_dir = "Machine_Learning/CS229/ps1/"

train_data = pd.read_csv(data_dir + "quasar_train.csv")
test_data = pd.read_csv(data_dir + "quasar_test.csv")

train_header = train_data.columns.values.astype(float).astype(int)
X = np.stack([np.ones(len(train_header)), train_header])
Y = train_data.values
Y_test = test_data.values
#X = np.full((len(Y[:, 0]), len(X), len(X[0])), X)

smoothed_y = np.asarray(smooth_values(X, Y))
smoothed_y_test = np.asarray(smooth_values(X, Y_test))

j = 0
plt.figure(1)
plt.plot(X[1, :], Y[j, :])
plt.xlabel("Wavelength (Angstrom)")
plt.ylabel("Luminous flux (lumens/m^2)")
plt.plot(X[1, :], smoothed_y[j])
plt.show()

estimates_train = []
errors_train = []
avg_train_error = 0

for i in range(smoothed_y.shape[0]):
    y_spec = smoothed_y[i]
    y_spec_left = y_spec[:right_index-1]
    y_spec_right = y_spec[right_index::]

    neighbours = np.asarray(find_spectra_neighbours(smoothed_y, y_spec_right))
    left_estimate = spectra_estimate(neighbours, y_spec_right)
    estimates_train.append(left_estimate)
    error = spectra_distance(left_estimate, y_spec_left)
    errors_train.append(error)
    avg_train_error += error

avg_train_error = avg_train_error/smoothed_y.shape[0]
print(avg_train_error)

for k in range(3):
    plt.figure(k)
    plt.plot(X[1, :], smoothed_y[k])
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Luminous flux (lumens/m^2)")
    plt.plot(X[1, :right_index-1], np.mean(estimates_train[k], axis=0))
    plt.show()
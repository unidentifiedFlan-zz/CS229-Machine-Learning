from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import random


def initialise_clusters(image, num_clusters):
    clusters = []
    for i in range(num_clusters):
        rand_x = random.randint(0, image.shape[1] - 1)
        rand_y = random.randint(0, image.shape[0] - 1)
        c = image[rand_x][rand_y]
        clusters.append(c)

    return clusters


def calculate_distance(image, cluster):
    diff = np.subtract(image, cluster)
    distances = np.linalg.norm(diff, axis=2)
    return distances


def calculate_means(image_with_assignments, num_clusters):
    means = np.zeros((num_clusters, 3))

    for c in range(num_clusters):
        mask = (image_with_assignments[:, :, 3] == c)
        masked_image = np.copy(image_with_assignments[:, :, 0:3])
        for i in range(3):
            masked_image[:, :, i] = mask * masked_image[:, :, i]

        count = np.count_nonzero(mask, axis=0)
        count = count.sum(axis=0)
        if count != 0:
            sum = masked_image.sum(axis=0)
            sum = sum.sum(axis=0)
            mean = (sum/count).astype(int)
            means[c, :] = mean

    return means


def assign_clusters(image, clusters):
    assigned_image = np.zeros((image.shape[0], image.shape[1], image.shape[2] + 1))
    assigned_image[:, :, 0:3] = image
    distances = np.zeros((image.shape[0], image.shape[1], len(clusters)))

    for i, c in enumerate(clusters):
        distances[:, :, i] = calculate_distance(image, c)

    assignments = np.argmin(distances, axis=2)
    assigned_image[:, :, 3] = assignments
    return assigned_image


def create_clustered_image(image_with_assignments, clusters):
    new_image = np.zeros((image_with_assignments.shape[0], image_with_assignments.shape[1], image_with_assignments.shape[2] - 1))
    for i, c in enumerate(clusters):
        original_mask = image_with_assignments[:, :, 3] == i
        for j in range(3):
            mask = np.copy(original_mask)*c[j]
            new_image[:, :, j] = new_image[:, :, j] + np.copy(mask)

    return new_image.astype(int)


def k_means_cluster(image, num_clusters):
    min_num_iterations = 30
    clusters = initialise_clusters(image, num_clusters)
    image_with_assignments = np.zeros((image.shape[0], image.shape[1], image.shape[2]+1))

    iteration = 0
    max_diff = 1000
    tolerance = 5
    while (iteration < min_num_iterations or max_diff > tolerance):
        image_with_assignments = assign_clusters(image, clusters)
        new_clusters = calculate_means(image_with_assignments, num_clusters)
        cluster_diffs = new_clusters - clusters
        max_diff = np.max(cluster_diffs)
        clusters = new_clusters
        iteration = iteration+1

    #return create_clustered_image(image_with_assignments, clusters).astype(int)
    return clusters


def compress_image(image, clusters):
    assigned_image = assign_clusters(image, clusters)
    return create_clustered_image(assigned_image, clusters)


def main():
    path = "Machine_Learning/CS229/ps3/"
    path_large = path + "mandrill-large.tiff"
    path_small = path + "mandrill-small.tiff"

    mandrill_large = imread(path_large)
    mandrill_small = imread(path_small)
    plt.imshow(mandrill_large)
    plt.show()
    small_image = np.array(mandrill_small)
    large_image = np.array(mandrill_large)
    num_clusters = 16
    clusters = k_means_cluster(small_image, num_clusters)
    compressed_mandrill = compress_image(large_image, clusters)
    plt.imshow(compressed_mandrill)
    plt.show()

if __name__ == '__main__':
    main()
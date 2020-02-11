import numpy as np
from scipy import spatial


def order_process(features):
    points = features[['x', 'y', 'r']].values
    orders = order_and_neighbors(points[:, :2])
    features['order_r_nearest_6'] = np.real(orders).astype('float32')
    features['order_i_nearest_6'] = np.imag(orders).astype('float32')
    return features


def order_and_neighbors(points):
    tree = spatial.cKDTree(points)
    dists, indices = tree.query(points, 7)
    neighbour_indices = indices[:, 1:]
    neighbour_positions = points[neighbour_indices, :]
    neighbour_vectors = neighbour_positions - points[:, np.newaxis, :]
    angles = np.angle(
        neighbour_vectors[:, :, 0] + 1j * neighbour_vectors[:, :, 1])
    steps = np.exp(6j * angles)
    orders = np.sum(steps, axis=1)
    orders /= 6
    return orders

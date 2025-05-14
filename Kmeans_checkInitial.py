import sys
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, data, k, a1, b1, a2, b2):
        self.data = data
        self.k = k
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

    def train(self, max_iter):
        centroids = KMeans.centroids_init(self.data, self.k, self.a1, self.b1, self.a2, self.b2)
        num_examples = [self.data.shape[0], self.data.shape[1]]
        closest_cent_ids = np.empty((num_examples[0],num_examples[1],1)).astype(np.uint8)

        for _ in range(max_iter):
            #distance of each pixel to the cent
            closest_cent_ids = KMeans.centroids_find_closest(self.data, centroids)
            #renew the cent
            centroids = KMeans.centroids_compute(self.data, closest_cent_ids, self.k)

        return centroids, closest_cent_ids

    @staticmethod
    def centroids_init(data, k, a1, b1, a2, b2):
        centroids = []
        centroids.append(data[a1][b1])
        print("a1,b1,", a1, b1)
        centroids.append(data[a2][b2])
        print("a2,b2,", a2, b2)
        centroids = np.asarray(centroids)
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        num_examples = [data.shape[0], data.shape[1]]
        dim = data.shape[2]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples[0],num_examples[1],1))
        for row_idx in range(num_examples[0]):
            for column_idx in range(num_examples[1]):
                distance = cdist(data[row_idx][column_idx].reshape(1, dim), centroids, metric='cosine')
                closest_centroids_ids[row_idx][column_idx] = np.argmin(distance)

        closest_centroids_ids=closest_centroids_ids.astype(np.uint8)
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_cent_ids, k):
        num_i = np.zeros((k,1))
        num_features = data.shape[2]
        centroids = np.zeros((k, num_features))
        #centroids = np.zeros((k, num_features)).astype(np.uint32)
        for centroid_id in range(k):
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if closest_cent_ids[i][j] == centroid_id:
                        num_i[centroid_id] += 1
                        centroids[centroid_id] += data[i][j]
        for centroid_id in range(k):
            centroids[centroid_id] = centroids[centroid_id] / num_i[centroid_id]

        return centroids
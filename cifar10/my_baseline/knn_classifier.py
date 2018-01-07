import numpy as np
import time


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            if i % 100 == 0:
                print("Computeing Dist", i)
            for j in range(num_train):
                dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            tic = time.time()
            cloned_array = np.array([X[i]] * num_train)
            toc = time.time()
            print("time", toc - tic)
            print("shape", self.X_train.shape)
            print("shape", cloned_array.shape)
            dists[i] = np.linalg.norm(self.X_train - cloned_array, axis=1)
        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        X_train_2 = self.X_train * self.X_train
        X_train_2 = np.sum(X_train_2, axis=1)

        X_train_2_repeat = np.array([X_train_2] * X.shape[0])

        X_2 = X * X
        X_2 = np.sum(X_2, axis=1)
        X_2_repeat = np.array([X_2] * self.X_train.shape[0]).transpose()

        X_dot_X_train = X.dot(self.X_train.T)
        dists = X_train_2_repeat + X_2_repeat - 2 * X_dot_X_train
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dists_i = dists[i]
            closest_y = self.y_train[dists_i.argsort()[:k]]
        return y_pred

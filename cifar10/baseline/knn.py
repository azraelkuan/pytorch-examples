import numpy as np
from data_utils import load_CIFAR10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


# loading data
cifar10_dir = "../data/cifar-10-batches-py"

X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# subsample the data to find the best k

num_training = 5000
mask = range(num_training)
x_train = X_train[mask]
y_train = Y_train[mask]

num_test = 500
mask = range(num_test)
x_test = X_test[mask]
y_test = Y_test[mask]

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print("Sample Data shape: x_train: {}, x_test: {}".format(x_train.shape, x_test.shape))


classifier = KNeighborsClassifier()
pipeline = Pipeline([("knn", classifier)])
param_grid = [
  {'knn__n_neighbors': [1, 3, 5, 7, 9, 10, 13, 17, 20, 50, 75, 100], 'knn__weights':['uniform', 'distance']},
 ]
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=20, cv=5)
grid_search.fit(x_train, y_train)

y_test_pred = grid_search.predict(x_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / len(y_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct, len(y_test), accuracy))


# # for cross validation
#
# num_folds = 5
# k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
#
#
# range_split = np.array_split(range(x_train.shape[0]), num_folds)
# y_train_folds = [y_train[range_split[i]] for i in range(num_folds)]
# x_train_folds = [x_train[range_split[i]] for i in range(num_folds)]
#
# k_to_accuracies = {}
#
# for k in k_choices:
#     for fold in range(num_folds): # This fold will be omitted.
#         # Creating validation data and temp training data
#         validation_x_test = x_train_folds[fold]
#         validation_y_test = y_train_folds[fold]
#         temp_x_train = np.concatenate(x_train_folds[:fold] + x_train_folds[fold + 1:])
#         temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])
#
#         # Initializing a class
#         test_classifier = KNeighborsClassifier(n_neighbors=k)
#         test_classifier.fit(temp_x_train, temp_y_train)
#
#         # Computing the distance
#         temp_y_test_pred = test_classifier.predict(validation_x_test)
#
#         # Checking accuracies
#         num_correct = np.sum(temp_y_test_pred == validation_y_test)
#         num_test = validation_x_test.shape[0]
#         accuracy = float(num_correct) / num_test
#         k_to_accuracies[k] = k_to_accuracies.get(k,[]) + [accuracy]
#
# for k in sorted(k_to_accuracies):
#     for accuracy in k_to_accuracies[k]:
#         print("k = %d, accuracy = %f" % (k, accuracy))
#
# accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
#
# best_k = k_choices[np.argmax(accuracies_mean)]
#
# classifier = KNeighborsClassifier(n_neighbors=best_k)
# classifier.fit(X_train, Y_train)
# Y_test_pred = classifier.predict(X_test)
#
# num_correct = np.sum(Y_test_pred == Y_test)
# accuracy = float(num_correct) / X_test.shape[0]
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


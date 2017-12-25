import numpy as np
import pandas as pd
import pickle
from data_utils import load_CIFAR10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


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

print("Begin to select best_parameters")
classifier = KNeighborsClassifier()
pipeline = Pipeline([("knn", classifier)])
param_grid = [
  {'knn__n_neighbors': [1, 3, 5, 7, 9, 10, 13, 17, 20, 50, 75, 100], 'knn__weights':['uniform', 'distance']},
 ]
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=20, cv=5)
grid_search.fit(x_train, y_train)

print("*"*50)
cv_results = pd.DataFrame(grid_search.cv_results_)
print(cv_results)
print("*"*50)

best_parameters = grid_search.best_estimator_.named_steps['knn']
print("best_parameters: {}".format(best_parameters))

print("*"*100)
print("begin final classifier")
final_classifier = KNeighborsClassifier(n_neighbors=best_parameters.n_neighbors, weights=best_parameters.weights, n_jobs=20)
final_classifier.fit(X_train, Y_train)

Y_test_pred = final_classifier.predict(X_test)

num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / len(Y_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct, len(Y_test), accuracy))

with open('model/knn.pickle', 'wb') as f:
    pickle.dump(final_classifier, f)



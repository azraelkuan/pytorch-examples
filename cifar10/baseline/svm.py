import numpy as np
from data_utils import load_CIFAR10
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


# loading data
cifar10_dir = "../data/cifar-10-batches-py"

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# subsample data for more dfficient code execution
# num_training = 5000
# mask = range(num_training)
# x_train = X_train[mask]
# y_train = y_train[mask]
# num_test = 500
# mask = range(num_test)
# x_test = X_test[mask]
# y_test = y_test[mask]
# the image data has three chanels
# the next two step shape the image size 32*32*3 to 3072*1
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


# define model
svm = SVC()
pipeline = Pipeline([("svm", svm)])
param_grid = [
  {'svm__C': [1, 10, 100, 1000], 'svm__kernel': ['linear']},
  {'svm__C': [1, 10, 100, 1000], 'svm__gamma': [0.001, 0.0001], 'svm__kernel': ['rbf']},
 ]

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=20)
grid_search.fit(X_train, y_train)


y_test_pred = grid_search.predict(X_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / len(y_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct, len(y_test), accuracy))

joblib.dump(grid_search, 'model/svm_grid_search.dmp', compress=3)




import numpy as np
from data_utils import load_CIFAR10
from knn_classifier import KNearestNeighbor


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

# sub sample
num_training = 5000
mask = range(num_training)
x_train = X_train[mask]
y_train = Y_train[mask]

num_test = 500
mask = range(num_test)
x_test = X_test[mask]
y_test = Y_test[mask]

# use sample data to choose the best parameter
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]


range_split = np.array_split(range(x_train.shape[0]), num_folds)
y_train_folds = [y_train[range_split[i]] for i in range(num_folds)]
x_train_folds = [x_train[range_split[i]] for i in range(num_folds)]

k_to_accuracies = {}

for k in k_choices:
    for fold in range(num_folds):  # This fold will be omitted.
        # Creating validation data and temp training data
        validation_x_test = x_train_folds[fold]
        validation_y_test = y_train_folds[fold]
        temp_x_train = np.concatenate(x_train_folds[:fold] + x_train_folds[fold + 1:])
        temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])

        # Initializing a class
        test_classifier = KNearestNeighbor()
        test_classifier.train(temp_x_train, temp_y_train)

        # Computing the distance
        temp_dists = test_classifier.compute_distances_two_loops(validation_x_test)
        temp_y_test_pred = test_classifier.predict_labels(temp_dists, k=k)

        # Checking accuracies
        num_correct = np.sum(temp_y_test_pred == validation_y_test)
        num_test = validation_x_test.shape[0]
        accuracy = float(num_correct) / num_test
        print("k=", k, "Fold=", fold, "Accuracy=", accuracy)
        k_to_accuracies[k] = k_to_accuracies.get(k, []) + [accuracy]


for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
best_k = k_choices[np.argmax(accuracies_mean)]

classifier = KNearestNeighbor()
classifier.train(X_train, Y_train)
Y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / len(Y_test)
print('Got %d / %d correct => accuracy: %f' % (num_correct, len(Y_test), accuracy))






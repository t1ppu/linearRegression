import numpy as np
import pandas as pd

# Cross validation function
def cross_validation(X, y, k):
    acc = []
    s = len(X)//k
    for i in range(k):
        # Split the data into training and test sets
        if i==0:
            X_train = X[s:]
            y_train = y[s:]
            X_test = X[0:s]
            y_test = y[0:s]
        elif i==k-1:
            X_train = X[0:(k-1)*s]
            y_train = y[0:(k-1)*s]
            X_test = X[(k-1)*s:]
            y_test = y[(k-1)*s:]
        else:
            X_train = X[np.r_[0:i*s, i*s+s:len(X)]]
            y_train = y[np.r_[0:i*s, i*s+s:len(X)]]
            X_test = X[i*s:(i+1)*s]
            y_test = y[i*s:(i+1)*s]

        # Train the model using linear regression
        X_train = np.c_[np.ones(len(X_train)), X_train]
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Predict on the test set (Cross-validation)
        X_test = np.c_[np.ones(len(X_test)), X_test]
        y_pred = X_test.dot(theta)

        # Calculate accuracy by finding the mean squared error
        y_pred = np.round(y_pred, 2)
        mse = np.mean((y_test - y_pred) ** 2)
        acc.append(1 - mse/np.var(y_test))
    return np.mean(acc)


# Loading the iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
dataset = pd.read_csv(url)

# Shuffling the dataset
dataset = dataset.sample(frac = 1)

# Splitting the dataset into X (features) and y (target)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Assigning numerical values to class labels
y = pd.DataFrame(y).replace({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})
y = y.to_numpy()

# K-Fold Cross validation
k = 5
mean_acc = cross_validation(X, y, k)

# Printing accuracy of the model
print("Accuracy: {:.2f}%".format(mean_acc*100))

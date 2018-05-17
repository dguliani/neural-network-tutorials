import numpy as np
import sklearn.datasets as datasets

from matplotlib import pyplot as plt
import seaborn as sns


def sigmoid(x):
    """ Sigmoid activation function for an array of neuron outputs.

    Args:
      x: Array/single element of neuron outputs before activation 

    Returns: 
      An array the same length as x with the sigmoid activation applied. 
    """
    return 1/(1+np.exp(-x))


def dsigmoid(y):
    """ Derivative of the sigmoid activation function with respect to the input x of the activation.

      Args: 
        y: Array/single element of neuron outputs after activation 

      Returns: 
        An array the same length as y representing the derivative with respect to x
    """
    return y*(1-y)


def mse(y, pred):
    """Mean squared error function comparing real target values with predictions.

    Args:
      y: List containing real target values 
      pred: List containing predicted target values (must have the same shape as y)

    Returns:
      Mean squared error between the two arrays
    """
    return np.mean(np.square(y - pred))


# Using scikit-learn's breast cancer dataset: classifying cell features as malignant or benign
bc = datasets.load_breast_cancer()

x = bc['data']  # array of features with shape: (n_samples, n_features)
y = bc['target']  # array of binary values for the two classes: (n_samples)

shuffle_ind = np.random.permutation(len(x))  # shuffle data for training
x = np.array(x[shuffle_ind])
# convert y to a column to make it more consistent with x
y = np.array(y[shuffle_ind])[np.newaxis].T

x /= np.max(x)  # linearly scaling the inputs to a range between 0 and 1

train_fraction = 0.75
train_idx = int(len(x)*train_fraction)

x_train = x[:train_idx]
y_train = y[:train_idx]

x_val = x[train_idx:]
y_val = y[train_idx:]

# Training a single layer neural network.
# initialize single layer weights randomly with mean 0: single neuron in layer
W = 2*np.random.random((np.array(x).shape[1], 1)) - 1

lr = 0.5  # Learning rate set through trial and error
history = {"loss": [], "val_loss": [], "acc": [],
           "val_acc": []}  # Metrics to track while training
epochs = 1000

for iter in range(epochs):
    # forward propagation
    l0 = x_train  # layer 0 output
    l1 = sigmoid(np.matmul(l0, W))  # layer 1 output

    l1_error = l1 - y_train  # Output layer error, (dE/do)

    l1_gradient = dsigmoid(l1)  # (do/dnetx)

    # (dE/do * do/dnetx = dE/dnetx)
    l1_delta = np.multiply(l1_error, l1_gradient)

    # (dnetx/dwij * dE/dnetx = dE/dwij)
    l1_weight_delta = np.matmul(l0.T, l1_delta)

    # update weights with a scaling factor of learning rate
    W -= l1_weight_delta * lr

    if iter % 100 == 0:
        # Recording metrics
        train_loss = mse(y_train, l1)
        train_acc = np.sum(y_train == np.around(l1))/len(x_train)
        history["loss"].append(train_loss)
        history["acc"].append(train_acc)

        val_pred = sigmoid(np.dot(x_val, W))
        val_loss = mse(y_val, val_pred)
        val_acc = np.sum(y_val == np.around(val_pred))/len(x_val)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print('Epoch: {}, Training Loss: {}, Validation Acc: {}'.format(
            iter, train_loss, val_acc))

val_pred = sigmoid(np.dot(x_val, W))
print("Validation loss: {}".format(mse(y_val, val_pred)))
print("Validation accuracy: {}".format(
    np.sum(y_val == np.around(val_pred))/len(x_val)))

# Plotting
sns.set()
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history['loss'], label="Training Loss")
plt.plot(history['val_loss'], label="Validation Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history['acc'], label="Training Accuracy")
plt.plot(history['val_acc'], label="Validation Accuracy")
plt.legend()

plt.show()

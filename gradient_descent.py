import numpy as np


def stochastic_gradient_descent(data, labels, gradloss,
                                learning_rate=1):
    """Calculate updates using stochastic gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    shuffler = np.random.permutation(len(labels))
    data = data[shuffler]
    labels = labels[shuffler]
    
    for i in range(len(labels)):        
        yield learning_rate * gradloss(data[i], [labels[i]])

def minibatch_gradient_descent(data, labels, gradloss,
                               batch_size=10, learning_rate=1):
    """Calculate updates using minibatch gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param batch_size: number of datapoints in each batch
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    N = len(labels)    
    shuffler = np.random.permutation(N)    
    data = data[shuffler]
    labels = labels[shuffler]
    
    i_s = 0
    while i_s < N:
        i_e = i_s + batch_size
        if i_e > N:
            i_e = N        
        yield learning_rate * gradloss(data[i_s:i_e], labels[i_s:i_e])        
        i_s = i_e

def batch_gradient_descent(data, labels, gradloss,
                           learning_rate=1):
    """Calculate updates using batch gradient descent algorithm

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param learning_rate: gradient scaling parameter
    :yield: yields scaled gradient
    """
    yield learning_rate * gradloss(data, labels)

def newton_raphson_method(data, labels, gradloss, hessianloss):
    """Calculate updates using Newton-Raphson update formula

    :param data: numpy array of shape [N, dims] representing N datapoints
    :param labels: numpy array of shape [N]
                   representing datapoints' labels/classes
    :param gradloss: function, that calculates the gradient of the loss
                     for the given array of datapoints and labels
    :param hessianloss: function, that calculates the Hessian matrix
                        of the loss for the given array of datapoints and labels
    :yield: yield once the update of Newton-Raphson formula
    """
    gradient = gradloss(data, labels)
    hessian = hessianloss(data, labels)
    yield np.linalg.inv(hessian) @ gradient

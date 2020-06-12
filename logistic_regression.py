import numpy as np

from gradient_descent import newton_raphson_method


class LogisticRegression:
    def __init__(self, dims,
                 update_method=newton_raphson_method,
                 update_params=None,
                 epochs=10):
        """Initialize Logistic Regression model

        :param dims: number of dimensions of data
        :param epochs: number of iterations over whole data
        :param update_method: update formula to use
        :param update_params: additional key word parameters to pass
                              to update function
        """
        if update_params is None:
            update_params = {}
        if update_method is newton_raphson_method:
            update_params['hessianloss'] = self.hessianloss
        self.dims = dims
        self.epochs = epochs
        self.update_method = update_method
        self.update_params = update_params
        self.w = self._generate_initial_weights(dims)

    @staticmethod
    def _generate_initial_weights(dims):        
        #return np.random.randn(dims)
        return np.zeros(dims)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, data, labels):
        """Fit the model and fix weight vector

        :param data: [N X dims] dimensional numpy array of floats
        :param labels: [N] dimensional numpy array of 1s and -1s denoting class
        :yield: the function will yield weight vector after each update
        """
        yield self.w
        for num_epoch in range(self.epochs):
            print(f"epoch N{num_epoch}:", end='\r', flush=True)
            # FIXME: Won't work correctly for windows, sorry :/
            for dw in self.update_method(data, labels, self.gradloss,
                                         **self.update_params):
                self.w -= dw
                yield self.w

    def loss(self, data, labels):
        """Calculate the loss of the model for current weights on the given data
        The loss function is equal to -log likelihood.
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        labels_hat = self.calculate_probabilities(data)
        return -(labels.dot(np.log(labels_hat)) + (1 - labels).dot(np.log(1 - labels_hat)))

    def gradloss(self, data, labels):
        """Calculate the gradient of loss
        
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        return data.T.dot(self.calculate_probabilities(data) - labels)
    
    def hessianloss(self, data, labels):
        """Calculate the Hessian matrix of loss
        :param data: [N, dims] dimensional numpy array of datapoints
        :param labels: [N] dimensional array of 1s and 0s
        """
        N = len(labels)
        D = np.zeros((N, N))
        for i in range(N):
            s_i = self.calculate_probabilities(data[i])
            D[i, i] = s_i * (1 - s_i)        
        return data.T.dot(D).dot(data)

    def calculate_probabilities(self, data):
        """Calculate probabilities for each datapoint of the given data
           of being from the first class

        :param data: [N X dims] dimensional numpy array to predict classes
        :return: numpy array of probabilities,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> np.all(model.calculate_probabilities(np.array([[-2, 1]]))
        ...        == np.array([0.5]))
        True
        >>> np.all(model.calculate_probabilities(np.array([[2, 0]]))
        ...        == model.calculate_probabilities(np.array([[0, 1]])))
        True
        """
        return self.sigmoid(data.dot(self.w.T))

    def predict(self, data):
        """Calculate labels for each datapoint of the given data

        :param data: [N X dims] dimensional numpy array to predict classes
        :return: numpy array of 1s and -1s,
                 where return_i denotes data_i's class
        >>> model = LogisticRegression(2)
        >>> model.w = np.array([1, 2])
        >>> np.all(model.predict(np.array([[2, 1], [1, 0], [0, -1]]))
        ...        == np.array([1, 1, 0]))
        True
        """
        return (self.calculate_probabilities(data) > 0.5).astype(int)


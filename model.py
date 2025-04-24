import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, nb_neurons_per_layer, activation_fct, learning_rate, loss_treshold, max_iteration):
        """
        Initialize neural network parameters.
        """
        np.random.seed(55)  # Ensure reproducibility

        self.nb_layers = len(nb_neurons_per_layer) - 1
        self.nb_neurons_per_layer = nb_neurons_per_layer
        self.activation_fct = activation_fct
        self.learning_rate = learning_rate
        self.loss_treshold = loss_treshold
        self.max_iteration = max_iteration

        # Weight and bias initialization
        self.weights = []
        self.biases = []

        for i in range(self.nb_layers):
            input_size = nb_neurons_per_layer[i]
            output_size = nb_neurons_per_layer[i + 1]

            self.biases.append(np.random.normal(0, 0.1, output_size))

            if activation_fct in ['sigmoid', 'tanh']:
                limit = np.sqrt(6 / (input_size + output_size))
                weight_matrix = np.random.uniform(-limit, limit, (output_size, input_size))
            elif activation_fct in ['relu', 'leaky_relu']:
                weight_matrix = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)

            self.weights.append(weight_matrix)

        # Intermediate variables for forward/backward pass
        self.z = [np.zeros((n, nb_neurons_per_layer[0])) for n in nb_neurons_per_layer[1:]]
        self.activations = [np.zeros((n,)) for n in nb_neurons_per_layer[1:]]

        self.dj_dw = [np.zeros_like(w) for w in self.weights]
        self.dj_db = [np.zeros(w.shape[0]) for w in self.weights]
        self.dj_dz = [np.zeros((n, nb_neurons_per_layer[0])) for n in nb_neurons_per_layer[1:]]

        self.loss = []
        self.test_loss = None

    def linear(self, i, X_train):
        """
        Compute the linear transformation z = W * x + b for layer i.
        """
        input_data = X_train.T if i == 0 else self.activations[i - 1]
        bias = self.biases[i][:, np.newaxis]
        self.z[i] = self.weights[i] @ input_data + bias

    def activation_function(self, i):
        """
        Apply the selected activation function to layer i.
        """
        if self.activation_fct == 'sigmoid':
            self.activations[i] = 1 / (1 + np.exp(-self.z[i]))
        elif self.activation_fct == 'relu':
            self.activations[i] = np.maximum(0, self.z[i])
        elif self.activation_fct == 'tanh':
            self.activations[i] = np.tanh(self.z[i])
        elif self.activation_fct == 'leaky_relu':
            self.activations[i] = np.where(self.z[i] > 0, self.z[i], 0.01 * self.z[i])

    def forwardPropagation(self, X_train):
        """
        Run a forward pass through the network.
        """
        for i in range(self.nb_layers):
            self.linear(i, X_train)
            self.activation_function(i)

    def compute_loss(self, y):
        """
        Compute and store mean squared error loss.
        """
        loss_vector = np.square(y.T - self.activations[-1]) / 2
        self.loss.append(np.mean(loss_vector))

    def backPropagation(self, X_train, y):
        """
        Compute gradients using backpropagation.
        """
        self.dj_dz[-1] = self.activations[-1] - y.T

        for i in reversed(range(self.nb_layers)):
            input_data = X_train if i == 0 else self.activations[i - 1].T
            self.dj_dw[i] = self.dj_dz[i] @ input_data
            self.dj_db[i] = np.mean(self.dj_dz[i], axis=1)

            if i > 0:
                dz = self.weights[i].T @ self.dj_dz[i]
                if self.activation_fct == 'sigmoid':
                    act = self.activations[i - 1]
                    self.dj_dz[i - 1] = dz * act * (1 - act)
                elif self.activation_fct == 'relu':
                    self.dj_dz[i - 1] = dz * (self.z[i - 1] > 0)
                elif self.activation_fct == 'tanh':
                    self.dj_dz[i - 1] = dz * (1 - np.tanh(self.z[i - 1]) ** 2)
                elif self.activation_fct == 'leaky_relu':
                    self.dj_dz[i - 1] = dz * np.where(self.z[i - 1] > 0, 1, 0.01)

    def update_params(self):
        """
        Update weights and biases using gradients.
        """
        for i in range(self.nb_layers):
            self.weights[i] -= self.learning_rate * self.dj_dw[i]
            self.biases[i] -= self.learning_rate * self.dj_db[i]

    def train(self, X, y, return_loss='True', verbose=True):
        """
        Train the network using the provided data.
        """
        i = 0
        while i <= self.max_iteration and (i == 0 or self.loss[-1] > self.loss_treshold):
            self.forwardPropagation(X)
            self.compute_loss(y)
            self.backPropagation(X, y)
            self.update_params()

            if verbose == True and i % 100 == 0:
                print(f'Epoch {i} : loss = {self.loss[-1]}')
            i += 1

        return self.loss if return_loss == 'True' else None

    def test(self, X, y, return_loss='True'):
        """
        Test the network and return predictions and/or loss.
        """
        self.forwardPropagation(X)
        self.compute_loss(y)
        self.test_loss = self.loss[-1]

        return (self.test_loss, self.activations[-1]) if return_loss == 'True' else self.activations[-1]

    def softmax(self, logits):
        """
        Compute softmax for logits (for classification tasks).
        """
        exp_logits = np.exp(logits - np.max(logits))  # Stability trick
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

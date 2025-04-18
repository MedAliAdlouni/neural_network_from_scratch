import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    def __init__(self, nb_neurons_per_layer, activation_fct, learning_rate, loss_treshold, max_iteration):
        # Initialize the neural network parameters, weights, biases, and various other attributes.
        np.random.seed(55)  # Set a seed for reproducibility
        self.nb_layers = len(nb_neurons_per_layer) - 1  # Calculate the number of layers
        self.nb_neurons_per_layer = nb_neurons_per_layer  # Store the number of neurons per layer
        self.activation_fct = activation_fct  # Set the activation function
        self.weights = []  # List to store weights for each layer
        self.biases = []  # List to store biases for each layer

        for i in range(len(self.nb_neurons_per_layer) - 1):
            input_size = self.nb_neurons_per_layer[i]  # Number of input neurons for this layer
            output_size = self.nb_neurons_per_layer[i + 1]  # Number of output neurons for this layer
            self.biases.append(np.random.normal(0, 0.1, output_size))  # Initialize biases

            # Xavier Initialization for Sigmoid and Tanh activation functions
            if self.activation_fct in ['sigmoid', 'tanh']:
                limit = np.sqrt(6 / (input_size + output_size))
                weight_matrix = np.random.uniform(-limit, limit, (output_size, input_size))
                self.weights.append(weight_matrix)

            # He Initialization for ReLU and Leaky ReLU activation functions
            if self.activation_fct in ['relu', 'leaky_relu']:
                weight_matrix = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
                self.weights.append(weight_matrix)

        # Initialize arrays to hold intermediate values
        self.z = [np.zeros((output_size, nb_neurons_per_layer[0])) for output_size in self.nb_neurons_per_layer[1:]]
        self.activations = [np.zeros((output_size,)) for output_size in self.nb_neurons_per_layer[1:]]
        self.loss = []  # List to track loss over iterations
        self.loss_treshold = loss_treshold  # Threshold for loss to stop training
        self.learning_rate = learning_rate  # Learning rate for weight updates
        self.dj_dw = np.array([np.zeros(weight.shape) for weight in self.weights], dtype='object')  # Gradient w.r.t weights
        self.dj_db = np.array([np.zeros(weight.shape[0]) for weight in self.weights], dtype='object')  # Gradient w.r.t biases
        self.dj_dz = [np.zeros((output_size, nb_neurons_per_layer[0])) for output_size in self.nb_neurons_per_layer[1:]]
        self.max_iteration = max_iteration  # Maximum iterations for training
        self.test_loss = None  # Variable to store test loss

    def linear(self, i, X_train):
        # Compute the linear transformation (z) for the i-th layer.
        if i == 0:
            bias = np.array(self.biases[i])[:, np.newaxis]  # Bias for the first layer
            self.z[i] = self.weights[i] @ X_train.T + bias  # First layer output
        else:
            bias = np.array(self.biases[i])[:, np.newaxis] # Bias for subsequent layers
            self.z[i] = self.weights[i] @ self.activations[i - 1] + bias  # Output from previous layer

    def activation_function(self, i):
        # Apply the activation function to the pre-activation values (z) for the i-th layer.
        if self.activation_fct == 'sigmoid':
            self.activations[i] = 1 / (1 + np.exp(-self.z[i]))  # Sigmoid activation
        elif self.activation_fct == 'relu':
            self.activations[i] = np.maximum(0, self.z[i])  # ReLU activation
        elif self.activation_fct == 'tanh':
            self.activations[i] = np.tanh(self.z[i])  # Tanh activation
        elif self.activation_fct == 'leaky_relu':
            self.activations[i] = np.where(self.z[i] > 0, self.z[i], 0.01 * self.z[i])  # Leaky ReLU activation

    def forwardPropagation(self, X_train):
        # Perform forward propagation through the network for a given input (X_train).
        for i in range(self.nb_layers):
            self.linear(i, X_train)  # Calculate linear transformation
            self.activation_function(i)  # Apply activation function

        
    def compute_loss(self, y):
        # Compute the loss between the predicted outputs and the true labels.
        list_loss_for_each_data_observation = np.square((y.T - self.activations[-1])) / 2  # Mean squared error loss
        avg_loss = np.mean(list_loss_for_each_data_observation)  # Average loss
        self.loss.append(avg_loss)  # Store loss for this iteration

    def backPropagation(self, X_train, y):
        # Perform backpropagation to compute gradients for weights and biases.
        self.dj_dz[-1] = (self.activations[-1] - y.T)  # Derivative of loss w.r.t output layer's activation
        for i in range(self.nb_layers - 1, -1, -1):
            if i != 0:
                self.dj_dw[i] = np.dot(self.dj_dz[i], self.activations[i - 1].T)  # Gradient w.r.t weights
                self.dj_db[i] = np.mean(self.dj_dz[i], axis=1)  # Gradient w.r.t biases
            else:
                self.dj_dw[i] = np.dot(self.dj_dz[i], X_train)  # Gradient w.r.t input layer weights
                self.dj_db[i] = np.mean(self.dj_dz[i], axis=1)  # Gradient w.r.t input layer biases

            if i > 0:
                # Compute gradient for previous layer based on the activation function
                if self.activation_fct == 'sigmoid':
                    self.dj_dz[i - 1] = np.dot(self.weights[i].T, self.dj_dz[i]) * self.activations[i - 1] * (1 - self.activations[i - 1]).astype(float)
                    
                elif self.activation_fct == 'relu':
                    print(f'weights[{i}].T.shape: {self.weights[i].T.shape}')
                    print(f'dj_dz[{i}].shape: {self.dj_dz[i].shape}')
                    print(f'z[{i}].shape: {self.z[i].shape}')
                    print(f'dj_dz[{i - 1}].shape before assignment: {self.dj_dz[i - 1].shape}')
                    print(f'self.z[{i-1}].shape : {self.z[i-1].shape}')
                    print(f'self.z[{i}].shape : {self.z[i].shape}')
                    print(np.dot(self.weights[i].T, self.dj_dz[i]).shape)
                    print((self.z[i-1] > 0).shape)
                    self.dj_dz[i - 1] = np.dot(self.weights[i].T, self.dj_dz[i]) * (self.z[i-1] > 0)
                elif self.activation_fct == 'tanh':
                    self.dj_dz[i - 1] = np.dot(self.weights[i].T, self.dj_dz[i]) * (1 - np.tanh(self.z[i]) ** 2)  # Derivative of tanh
                elif self.activation_fct == 'leaky_relu':
                    self.dj_dz[i - 1] = np.dot(self.weights[i].T, self.dj_dz[i]) * np.where(self.z[i] > 0, 1, 0.01)  # Derivative of Leaky ReLU

    def update_params(self):
        # Update weights and biases using computed gradients.
        for i in range(self.nb_layers):
            self.weights[i] = self.weights[i] - self.learning_rate * self.dj_dw[i]  # Update weights
            self.biases[i] = self.biases[i] - self.learning_rate * self.dj_db[i]  # Update biases

    def train(self, X, y, return_loss='True'):
        # Train the neural network using the provided training data (X, y).
        i = 0
        while (i <= self.max_iteration) and (i == 0 or (self.loss and self.loss[-1] > self.loss_treshold)):
            # Loop: Forward pass + loss calculation + Backward pass + parameters update
            self.forwardPropagation(X)  # Forward propagation
            self.compute_loss(y)  # Compute loss
            self.backPropagation(X, y)  # Backward propagation
            self.update_params()  # Update parameters

            # Print loss every 100 epochs
            if i % 100 == 0: 
                print(f'Epoch {i} : loss = {self.loss[-1]}')

            i += 1
            
        if return_loss == 'True':
            return self.loss  # Return loss history if requested

    def test(self, X, y, return_loss='True'):
        # Test the neural network on the provided data (X, y) and compute test loss.
        self.forwardPropagation(X)  # Forward propagation for testing
        self.compute_loss(y)  # Compute test loss

        self.test_loss = self.loss[-1]  # Store the last computed loss as test loss

        # Return test loss and predictions
        if return_loss == 'True':
            return self.test_loss, self.activations[-1]  # Return test loss and the predictions
        else:
            return self.activations[-1]  # Return only the predictions if loss is not requested

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Neural Network Implementation with Backpropagation


## Description
This project implements a multilayer perceptron (MLP) from scratch using NumPy, focusing on the backpropagation algorithm for training the network. The primary goal is to classify binary and multi-class datasets, with a particular focus on the XOR problem. Different hyperparameters will be tested

## Features
- Implementation of a multilayer perceptron (MLP)
- Forward and backward propagation algorithms
- Support for ReLU and softmax activation functions
- Ability to train on binary classification tasks (e.g., XOR)
- Customizable hyperparameters (learning rate, number of neurons, etc.)

## 🧪 Running the Code

1. **Clone the repo**  
   ```bash
   git clone https://github.com/MedAliAdlouni/neural_network_from_scratch
   cd neural_network_from_scratch
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```
3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the notebook**

    - `main.ipynb` to run all experiments



## Class Definition

### `class NeuralNetwork`
The `NeuralNetwork` class implements a feedforward neural network. It allows for :

- Customizable architecture (number layers, number neurons, number of outputs, etc)
- Activation functions : sigmoid, hyperbolic tangent, ReLU and leaky ReLU 
- Learning rates, number of epochs, loss treshold, etc.

#### `__init__(self, nb_neurons_per_layer, activation_fct, learning_rate, loss_treshold, max_iteration)`
- **Purpose**: Initializes the neural network parameters and creates numpy arrays to stock pre-activations, activations, gradients, etc

- **Parameters**:
  - `nb_neurons_per_layer`: A list containing the number of neurons in each layer.
  - `activation_fct`: The activation function to use (e.g., 'sigmoid', 'relu', etc.).
  - `learning_rate`: The rate at which the model learns.
  - `loss_treshold`: The threshold for loss to stop training.
  - `max_iteration`: The maximum number of iterations for training.
  
- **Functionality**:
  - Sets the random seed for reproducibility.
  - Sets weights and biases initialisation. Proper initialization of weights is essential for neural network training because it sets the starting point for the optimization process, which significantly impacts convergence speed and the likelihood of reaching a good solution. Poor initialization can cause:
  
- **Vanishing gradients**: When weights are too small, gradients can become too small to influence weight updates, stalling learning.
- **Exploding gradients**: When weights are too large, gradients can grow excessively, leading to unstable updates and divergence.

In this code, two popular initialization methods are used based on the activation function:

1. **Xavier Initialization** (used for **Sigmoid** and **Tanh** activation functions):
   - **Formula**: For weights in a layer with `input_size` neurons feeding into `output_size` neurons, Xavier initialization sets the weights by sampling from a uniform distribution:
     $$
     W \sim \text{Uniform}\left(-\sqrt{\frac{6}{\text{input\_size} + \text{output\_size}}}, \sqrt{\frac{6}{\text{input\_size} + \text{output\_size}}}\right)
     $$
   - **Purpose**: This method keeps the variance of activations and gradients roughly the same across layers, which helps avoid the vanishing/exploding gradient problem in networks with Sigmoid and Tanh activations.

2. **He Initialization** (used for **ReLU** and **Leaky ReLU** activation functions):
   - **Formula**: For a layer with `input_size` neurons feeding into `output_size` neurons, He initialization samples weights from a normal distribution:
     $$
     W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{input\_size}}}\right)
     $$
   - **Purpose**: He initialization is optimized for ReLU-type activations, which are more sensitive to the magnitude of weights. By scaling the weights according to the number of input neurons, this method helps maintain stable gradients during training, which is crucial for efficient learning with ReLU and Leaky ReLU functions.

---

### `linear(self, i, X_train)`
- **Purpose**: Computes the linear transformation for the `i`-th layer.
- **Parameters**:
  - `i`: The index of the layer for which to compute the linear transformation.
  - `X_train`: The input data.
- **Functionality**: Calculates the pre-activation values (`z`) for the specified layer, taking into account the input from the previous layer.

In a neural network, the linear transformation for a layer can be represented by the equation:

$$
z^{(i)} = W^{(i)} \cdot a^{(i-1)} + b^{(i)}
$$

where:

$$
\begin{align*}
& z^{(i)} \text{ is the pre-activation value for the } i \text{-th layer,} \\
& W^{(i)} \text{ is the weight matrix connecting the } (i-1) \text{-th layer to the } i \text{-th layer,} \\
& a^{(i-1)} \text{ is the activation from the previous layer (or the input data } X_{\text{train}} \text{ if } i = 0), \\
& b^{(i)} \text{ is the bias vector for the } i \text{-th layer.}
\end{align*}
$$


This linear transformation prepares the data for the subsequent activation function applied in each layer, which adds non-linearity and enables the network to model complex patterns.

---

### `activation_function(self, i)`
- **Purpose**: Applies the activation function to the pre-activation values for the `i`-th layer.
- **Parameters**:
  - `i`: The index of the layer to apply the activation function to.
- **Functionality**: Transforms the linear output (`z`) using the chosen activation function (sigmoid, ReLU, tanh, or leaky ReLU) and stores the result in the `activations` array.

#### Equations:
- **Sigmoid Activation**:
  $$ a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1 + e^{-z^{(i)}}} $$
  
- **ReLU Activation**:
  $$ a^{(i)} = \text{ReLU}(z^{(i)}) = \max(0, z^{(i)}) $$
  
- **Tanh Activation**:
  $$ a^{(i)} = \tanh(z^{(i)}) = \frac{e^{z^{(i)}} - e^{-z^{(i)}}}{e^{z^{(i)}} + e^{-z^{(i)}}} $$
  
- **Leaky ReLU Activation**:
  $$ a^{(i)} = \text{Leaky ReLU}(z^{(i)}) = \begin{cases} 
  z^{(i)} & \text{if } z^{(i)} > 0 \\ 
  0.01 z^{(i)} & \text{otherwise} 
  \end{cases} $$

---

### `forwardPropagation(self, X_train)`
- **Purpose**: Performs forward propagation through the network for a given input.
- **Parameters**:
  - `X_train`: The training input data.
- **Functionality**: Iteratively calls the `linear` and `activation_function` methods for each layer to compute the output of the network.

#### Equations:
- **Forward Propagation**:
  $$ a^{(i)} = \phi(z^{(i)}) \quad \text{for each layer } i $$
  where \( \phi \) is the activation function used in that layer.

---

### `compute_loss(self, y)`
- **Purpose**: Computes the loss between the predicted outputs and the true labels.
- **Parameters**:
  - `y`: The true output values.
- **Functionality**: Calculates the mean squared error loss and stores it for tracking the training progress.

#### Equations:
- **Mean Squared Error Loss**:
  $$ L = \frac{1}{n} \sum_{j=1}^{n} \frac{1}{2}(y_j - \hat{y}_j)^2 $$
  where \( \hat{y}_j \) is the predicted output and \( n \) is the number of observations.

---

### `backPropagation(self, X_train, y)`
- **Purpose**: Performs backpropagation to compute gradients for weights and biases.
- **Parameters**:
  - `X_train`: The training input data.
  - `y`: The true output values.
- **Functionality**: Computes the gradients with respect to the weights and biases by iterating backward through the network, updating the derivatives based on the chosen activation function.

#### Equations:
- **Gradient of Loss with respect to Activation**:
  $$ \delta^{(i)} = \nabla_a L \cdot \phi'(z^{(i)}) $$
  where \( \nabla_a L \) is the derivative of the loss with respect to the activations, and \( \phi' \) is the derivative of the activation function.

---

### `update_params(self)`
- **Purpose**: Updates the weights and biases using the computed gradients.
- **Functionality**: Applies the gradient descent update rule to adjust the weights and biases based on the learning rate.

#### Equations:
- **Weight Update Rule**:
  $$ W^{(i)} \leftarrow W^{(i)} - \eta \frac{\partial L}{\partial W^{(i)}} $$

- **Bias Update Rule**:
  $$ b^{(i)} \leftarrow b^{(i)} - \eta \frac{\partial L}{\partial b^{(i)}} $$
  where \( \eta \) is the learning rate.

---

### `train(self, X, y, return_loss='True')`
- **Purpose**: Trains the neural network using the provided training data.
- **Parameters**:
  - `X`: The input training data.
  - `y`: The target output values.
  - `return_loss`: Boolean to indicate whether to return the loss history.
- **Functionality**: Repeatedly performs forward propagation, loss computation, backpropagation, and parameter updates until the maximum iterations are reached or the loss falls below the specified threshold. Prints the loss every 100 epochs.

---

### `test(self, X, y, return_loss='True')`
- **Purpose**: Tests the neural network on the provided data and computes the test loss.
- **Parameters**:
  - `X`: The input testing data.
  - `y`: The target output values.
  - `return_loss`: Boolean to indicate whether to return the test loss.
- **Functionality**: Performs forward propagation and computes the test loss, storing the last computed loss for further analysis.


## Getting Started

### Prerequisites
- Python 3.x
- NumPy library

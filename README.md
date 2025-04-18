# Neural Network Implementation with Backpropagation


## Description
This project implements a multilayer perceptron (MLP) from scratch using NumPy, focusing on the backpropagation algorithm for training the network. The primary goal is to classify binary and multi-class datasets, with a particular focus on the XOR problem. Different hyperparameters will be tested

## Features
- Implementation of a multilayer perceptron (MLP)
- Forward and backward propagation algorithms
- Support for ReLU and softmax activation functions
- Ability to train on binary classification tasks (e.g., XOR)
- Customizable hyperparameters (learning rate, number of neurons, etc.)



# Introduction to Neural Networks

This project is done by Mohammed Ali El Adlouni and is part of the Deep Learning course in the Master's program MALIA at Université Lyon 2.

In this project, the goal is to write the entire backpropagation algorithm from scratch using only Numerical Python package (numpy) and implement a simple multilayer perceptron (MLP). This will involve developing the necessary components for forward propagation, loss computation, and backpropagation to update the network's weights and biases effectively. We will also study the effect of some hyperparameters (learning rate, choice of activation function, etc) on the performance score and the elapsed time.

## 1. Architecture of Neural Networks

A neural network consists of layers of interconnected nodes, or neurons. Each neuron receives input, applies a linear transformation (using weights and biases), and then applies a non-linear activation function. The general structure can be represented as follows:

### Figure 1: Basic Neural Network Architecture


### Mathematical Representation

The output \( z \) of a neuron can be described by the equation:

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

Where:
- \(\hat{y}\) is the predicted output,  
- \(f\) is the activation function,  
- \(\mathbf{W}\) represents the weights, and  
- \(\mathbf{b}\) is the bias.



After calculating \( z \), a non-linear activation function is applied to produce the activation \( a \):

$$
a = \sigma(z)
$$

Common activation functions include:

- **Sigmoid**: 
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
  
- **ReLU (Rectified Linear Unit)**:
$$
\sigma(z) = \max(0, z)
$$

- **Tanh (Hyperbolic Tangent)**:
$$
\sigma(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

## 2. Forward Propagation

Forward propagation is the process of passing inputs through the network to obtain the output. For a multi-layer neural network, the output can be expressed as:

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})
$$

Where:
- \( \mathbf{a}^{(l)} \) is the activation of layer \( l \),
- \( \mathbf{W}^{(l)} \) and \( \mathbf{b}^{(l)} \) are the weights and biases for layer \( l \).

## 3. Loss Function

The loss function quantifies how well the neural network's predictions match the true labels. For regression tasks, the Mean Squared Error (MSE) is commonly used:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:
- \( N \) is the number of observations,
- \( y_i \) is the true value,
- \( \hat{y}_i \) is the predicted value.

## 4. Backpropagation

Backpropagation is an algorithm used to update the weights and biases of the neural network based on the computed loss. The gradient of the loss function with respect to the weights and biases is calculated to minimize the loss.

### Gradient Descent Update Rule

The weights and biases are updated using the following rules:

$$
\mathbf{W}^{(l)} \gets \mathbf{W}^{(l)} - \eta \frac{\partial \text{Loss}}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)} \gets \mathbf{b}^{(l)} - \eta \frac{\partial \text{Loss}}{\partial \mathbf{b}^{(l)}}
$$

Where \( \eta \) is the learning rate.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy library

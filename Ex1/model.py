import numpy as np
from utils import *

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, initialize = 'normal', activation = 'relu', l2=False, l2_lambda=0.0):

        self.initialize = initialize
        self.l2 = l2
        self.l2_lambda = l2_lambda

        if self.initialize == 'normal':
            self.W = np.random.randn(input_dim, output_dim) * 0.1
        elif self.initialize == 'xavier':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        elif self.initialize == 'he':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            raise ValueError("Initialize Type not supported.")

        self.b = np.zeros((1, output_dim))
        self.activation = activation
        self.X = None # input for layer
        self.A = None # output for layer
        self.Z = None # output after activation

        self.v_W = np.zeros_like(self.W) # momentum for weights
        self.v_b = np.zeros_like(self.b) # momentum for biases
        self.m_W = np.zeros_like(self.W) # Adam second moment for weights
        self.m_b = np.zeros_like(self.b) # Adam second moment for biases

        self.beta1 = 0.9 # momentum hyperparameter
        self.beta2 = 0.999 # Adam hyperparameter


    def forward(self, X): # when X contains only a mini-batch, it is ...
        # A = W * X + b
        # Z = activation(A)
        self.X = X  # Store input for backpropagation
        A = np.dot(X, self.W) + self.b
        self.A = A  # Store pre-activation for backpropagation

        if self.activation == 'relu':
            Z = relu(A)
        elif self.activation == 'leaky_relu':
            Z = leaky_relu(A)
        elif self.activation == 'tanh':
            Z = tanh(A)
        elif self.activation == 'sigmoid':
            Z = sigmoid(A)
        elif self.activation == 'linear':
            Z = A
        else:
            raise ValueError("Activation Type not supported.")
        self.Z = Z  # Store post-activation for backpropagation
        return Z

    def backward(self, dZ):
        # dW = dZ * dZ_dA * dA_dX
        if self.activation == 'relu':
            dA = dZ * relu_differential(self.A)
        elif self.activation == 'leaky_relu':
            dA = dZ * leaky_relu_differential(self.A)
        elif self.activation == 'tanh':
            dA = dZ * tanh_differential(self.A)
        elif self.activation == 'sigmoid':
            dA = dZ * sigmoid_differential(self.A)
        elif self.activation == 'linear':
            dA = dZ
        else:
            raise ValueError("Activation Type not supported.")

        m = self.X.shape[0]
        dW = np.dot(self.X.T, dA) / m
        db = np.sum(dA, axis=0, keepdims=True) / m
        dX = np.dot(dA, self.W.T)

        if self.l2:
            dW += l2_regularization_gradient(self.W, self.l2_lambda)
            
        # Update momentum and Adam parameters
        self.v_W = self.beta1 * self.v_W + (1 - self.beta1) * dW
        self.v_b = self.beta1 * self.v_b + (1 - self.beta1) * db

        self.m_W = self.beta2 * self.m_W + (1 - self.beta2) * (dW ** 2)
        self.m_b = self.beta2 * self.m_b + (1 - self.beta2) * (db ** 2)

        return dW, db, dX
    
class Model:
    def __init__(self, learning_rate=0.01, optimizer='sgd'):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def add_layer(self, layer_input_dim, layer_output_dim, initialize='xavier', activation='relu', l2=False, l2_lambda=0.0):
        self.layers.append(FullyConnectedLayer(layer_input_dim, layer_output_dim, initialize, activation, l2, l2_lambda))

    def forward(self, input_X):
        X = input_X
        for layer in self.layers:
            X = layer.forward(X)
        
        return X # 最终输出

    def backward(self, ypred_minus_ytrue, optimizer='sgd'): # dZ is the initial grad, (y_pred - y_true)
        dz = ypred_minus_ytrue
        for layer in reversed(self.layers):
            dW, db, dz = layer.backward(dz) # the input_grad for the big_num layer is the output_grad for the small_num layer
            if optimizer == 'sgd':
                layer.W -= self.learning_rate * dW
                layer.b -= self.learning_rate * db
            elif optimizer == 'momentum':
                layer.W -= self.learning_rate * layer.v_W
                layer.b -= self.learning_rate * layer.v_b
            elif optimizer == 'adam':
                m_W_corrected = layer.m_W / (1 - layer.beta2)
                m_b_corrected = layer.m_b / (1 - layer.beta2)
                layer.W -= self.learning_rate * m_W_corrected / (np.sqrt(layer.v_W) + 1e-8)
                layer.b -= self.learning_rate * m_b_corrected / (np.sqrt(layer.v_b) + 1e-8)

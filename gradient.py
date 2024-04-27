import numpy as np
from neuralnet import Neuralnetwork
import copy


def check_grad(model, x_train, y_train, epsilon=1e-2):
    # Forward and backward pass to compute gradients by backpropagation
    model.forward(x_train, y_train)
    model.backward(gradReqd = False)

    # Define the indices of weights to check
    outputlayer = model.layers[-1].w.shape
    hidden = model.layers[0].w.shape
    np.random.seed(42)
    weights = [
      # Output layer, last bias weight
      (-1, outputlayer[0] - 1, outputlayer[1] - 1),
      # First hidden layer, last bias weight
      (0, hidden[0] - 1, hidden[1] - 1),
      # Two weights from hidden-to-output layer
      (-1, np.random.randint(0, outputlayer[0] - 1), np.random.randint(0, outputlayer[1] - 1)),
      (-1, np.random.randint(0, outputlayer[0] - 1), np.random.randint(0, outputlayer[1] - 1)),
      # Two weights from input-to-hidden layer
      (0, np.random.randint(0, hidden[0] - 1), np.random.randint(0, hidden[1] - 1)),
      (0, np.random.randint(0, hidden[0] - 1), np.random.randint(0, hidden[1] - 1)),
    ]

    for layer, row, col in weights:
        # Create two copies of the model
        model_1 = copy.deepcopy(model)
        model_2 = copy.deepcopy(model)

        model_1.layers[layer].w[row, col] += epsilon
        loss_1 = model_1.loss(model_1.forward(x_train), y_train)

        model_2.layers[layer].w[row, col] -= epsilon
        loss_2 = model_2.loss(model_2.forward(x_train), y_train)

        numerical_grad = model.layers[-1].out_units*(loss_1 - loss_2) / (2 * epsilon)

        backprop_grad = model.layers[layer].gradient[row, col]

        diff = abs(numerical_grad - backprop_grad)
        
        print(f"Layer {layer}, Weight ({row}, {col}): Numerical: {numerical_grad}, Backprop: {backprop_grad}, diff: {diff}")


def checkGradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
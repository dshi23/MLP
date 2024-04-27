import numpy as np
import util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(x,0)

    def output(self, x):
        """
        TODO: Implement softmax function here.
        Remember to take care of the overflow condition (i.e. how to avoid denominator becoming zero).
        """
        num = np.exp(x)
        denum = num.sum(axis = 1, keepdims = True) + 0.0001
        return num / denum
        
    def grad_sigmoid(self, x):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return np.exp(-x)/(1+np.exp(-x))**2

    def grad_tanh(self, x):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1-self.tanh(x)**2
    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return (x > 0).astype(float)
    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1  #Deliberately returning 1 for output layer case


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        # Randomly initialize weights
        self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation = activation
        self.delta = None
        self.acc_gradient = self.w * 0
        self.out_units = out_units
        self.gradient = None
        
    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = util.append_bias(x)
        self.a = np.dot(self.w.T,self.x.T).T
        self.z = self.activation(self.a)
        return self.z
        
    def backward(self, deltaCur, learning_rate, momentum_gamma, L2, L1, gradReqd=True):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass

        When implementing softmax regression part, just focus on implementing the single-layer case first.
        """
        # activation_derivative = self.activation.backward(self.a)
        # if self.activation.activation_type == "output":
        #     delta = deltaCur 
        # else:
        #     print(deltaCur.shape,self.w.shape)
        #     delta = np.dot(deltaCur.T, self.w) * self.activation.backward(self.a)
        
        # self.dw = self.x.T @ delta
        # if gradReqd:
        #     self.w += learning_rate*self.dw
        # regularization
        # self.dw += regularization * self.w

        self.delta = self.activation.backward(self.a) * deltaCur[:, :self.out_units]
             
        batch_size = self.x.shape[0]
        self.gradient = -((self.x.T @ self.delta) / batch_size + float(L2) * self.w + float(L1) * ((self.w > 0)*1.0))
        gradient = (-1) * learning_rate * self.gradient
    
        self.acc_gradient = momentum_gamma * self.acc_gradient + (1-momentum_gamma) * gradient
        
        if gradReqd:
            self.w += self.acc_gradient

        deltaToPass = self.delta @ self.w.T
        return deltaToPass
        
        
        
class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.config = config
        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        self.targets = targets
        for i in range(self.num_layers):
            self.x = (self.layers[i])(self.x)
        self.y = self.x
        if targets is None:
            return self.y
        else:
            return self.loss(np.log(self.y+ 1e-8),self.targets), util.calculateCorrect(self.y,targets)/targets.shape[0]
        

    def loss(self, logits, targets):
        '''
        TODO: Compute the categorical cross-entropy loss and return it.
        '''
        E = -1/(logits.shape[0])* np.sum(targets*logits)
        return E

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        if self.config['momentum'] == False:
            momentum = 0
        else:
            momentum = self.config['momentum_gamma']
        
        delta = self.targets-self.y
        backward_layers = self.layers[::-1]
        
        for layer in backward_layers:
            delta = layer.backward(delta, 
                                    self.config['learning_rate'], 
                                    momentum, 
                                    self.config['L2_penalty'],
                                    self.config['L1_penalty'],
                                    gradReqd)



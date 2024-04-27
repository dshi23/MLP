import copy
from neuralnet import *
from util import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    # Read in the esssential configs
    val_losses = []
    val_accs = []
    train_losses = []
    train_accs = []
    
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    early_stop = config["early_stop"]
    early_stop_epoch = config["early_stop_epoch"]
    
    patience = early_stop_epoch
    best_model = None
    best_loss = float("inf")
    
    #SGD
    for epoch in range(epochs):
        batches = generate_minibatches(x_train, y_train, batch_size)
        
        for X, y in batches:
            model(X,y)
            model.backward()
        train_loss, train_acc = model(x_train, y_train)
        val_loss, val_acc = model(x_valid,y_valid)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if early_stop:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                patience = 0
                earlyStop =epoch
            else:
                patience += 1
                best_loss = val_loss
                if patience >= early_stop_epoch:
                    print("Early Stopped at {}".format(earlyStop))
                    model = best_model
                    #earlyStop = epoch
                    break
                
        print("Epoch: {}/{}, Training_loss: {}, validation_loss: {}, Validation_acc: {}".format(epoch, epochs, train_loss, val_loss, val_acc))
         
    return model, train_losses, train_accs, val_losses, val_accs, earlyStop

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    test_loss, test_acc = model(X_test,y_test)
    return test_acc, test_loss
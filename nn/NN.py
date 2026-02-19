import numpy as np  
import time
import os
import sys

from nn.utils import *

"""
This script defines a general NeuralNetwork class. We do it in a general way so we have a flexible model with an easy way of tracking errors. See the README file to more information of how this class works.

Example of ussage: 
"""
# TODO - test
class NeuralNetwork:

    # Define which Loss and Activations functions supports our Neural Network
    LossFunctions = ["CrossEntropy", "MSE"]
    ActivationFunctions = ["ReLU", "Sigmoind", "Softmax"]
    
    AF = {"ReLU": ReLU(), "Sigmoind": Sigmoind(), "Softmax": Softmax()}

    # DONE TODO - test
    def __init__(self, dimensions, activations, name = "NeuralNetwork"):
        """
        Parameters
        ----------
            · dimensions: list of the dimension of each hidden layer
            · activations: list of the activation function that must be applied after each layer.

            Note that the len of activations should be 1 less than the dimensions
        """
        self.name = name
        self.n_layers = len(dimensions) - 2 # Number oof hidden layers
        self.n_in = dimensions[0]
        self.n_out = dimensions[-1]

        self.layers = Sequence([])
        self.dimensions = dimensions
        self.activations = activations
     
        if self.n_layers == 0: # We don't have hidden layers
            self.layers.add(LinearLayer(self.n_in, self.n_out, activation=activations[0]))

        else:
            self.layers.add(LinearLayer(dimensions[0], dimensions[1],activation=activations[0]))
            for i in range(self.n_layers):
                if activations[i+1] not in NeuralNetwork.ActivationFunctions:
                        raise TypeError(f"Please, use some of the \033[1;31mavailable activation funcions: {NeuralNetwork.ActivationFunctions}\033[0m")
                
                self.layers.add(LinearLayer(dimensions[i+1],dimensions[i+2],activation=activations[i+1]))

            # Create last layer

        self.n_trainable_parameters = 0
        for i in range(len(self.layers)): self.n_trainable_parameters += self.layers[i].n_trainable_parameters
    
    # DONE TODO - test
    def forward(self, input):
        return self.layers.forward(input)

    # DONE TODO - test
    def train(self, data, epochs, learning_rate=0.1, loss=CrossEntropy(), print_training = True):
        """
        Parameters
        ----------
            · data: data must be a list with touples of (np.array(size=(n,)), label), where the labels should be a number between 0 and the number of possible labels minus 1.
            · epochs: the number of iterations that we apply the backpropagation.
            · learning_rate: stepsize of the backpropagation algorithm.
            · loss: the loss functions that we use to train the model.
        
        Returns
        -------
            Returns the same neural networks with the parameters updated. 
        """

        if str(loss) not in NeuralNetwork.LossFunctions: 
            raise TypeError(f"Please use some of the \033[1;31mavailable loss functions: {NeuralNetwork.LossFunctions}\033[0m")

        print(f"\033[1;33mTrainable Parameters:\033[0m\t{self.n_trainable_parameters}")
        print(f"\033[1;32mNumber of Epochs:\033[0m\t{epochs}\n")
        
        total = len(data)
        bar_width = 40

        times_epochs = []
        mean_loss_epochs = []

        start_training_model = time.time()
        sys.stdout.write(f"\033[1mTraining {self.name}:\t{'':40s}\n")

        for e in range(epochs):

            loss_epoch = []

            i = 0
            start_epoch = time.time()
            sys.stdout.write("\033[1A")
            p1 = int(bar_width * (e+1) / epochs)
            perc1 = int(100*p1/bar_width)
            sys.stdout.write(f"\r\033[1mTraining {self.name}:\t|\033[31;7m{' '*p1}\033[0m{' '*(bar_width-p1)}| {perc1:02d}%\n")
            sys.stdout.flush()
            sys.stdout.write(f"\r\tEpoch: {e+1:02d}\t|{'':40s}|")
            sys.stdout.flush()
            for X in data:
                i+=1
                x, y = X[0], X[1]
                p2 = int(bar_width * i / total)
                perc2 = int(100*p2/bar_width)
                sys.stdout.write(f"\r\tEpoch {e+1:02d}:\t|\033[32;7m{' '*p2}\033[0m{' '*(bar_width-p2)}| {perc2:02d}%")
                sys.stdout.flush()
                pred = self.forward(x)
                y = self.to_one_hot(y)
                
                loss_epoch.append(loss(pred,y))
                if str(loss) == "CrossEntropy" and str(self.layers[-1].activation) == "Softmax":
                    self.layers.backpropagate(step=learning_rate, delta = pred-y, update_parameters=True, first_delta_computed=True)
                else:
                    d = loss.partial(pred,y)
                    self.layers.backpropagate(step=learning_rate,delta=d, update_parameters=True, first_delta_computed= False)
            
            mean_loss_epochs.append(np.mean(loss_epoch))
            
            time_epoch = time.time() - start_epoch
            times_epochs.append(time_epoch)
        
        return mean_loss_epochs, times_epochs
               

    # DONE TODO - test
    def test(self, data):
        """
        Parameters
        ----------
            · data: data to test the model
        Returns
        -------
            · acc: the accuracy of the model for the data.
        """
        n = len(data)
        acc = 0

        for i in range(n):
            y = data[i][1]
            pred = np.argmax(self.forward(data[i][0]))
            acc +=  pred == y
        return acc/n
    
    # DONE 
    def to_one_hot(self, label):
        v = np.zeros(shape = (self.n_out,))
        v[label] = 1
        return v

    # DONE 
    def __str__(self):
        s = "Neural Network\n"
        ml = len(s)
        s += '-'*(len(s)-1) + '\n'
        
        s += f'>\033[1;32m Number of Hidden Layers: \033[0m ({self.n_layers})\n'
        s += f'>\033[1;33m Pass of Information: \033[0m\n'
        for l in self.layers:
            s += '\t'+ str(l) + '\n'
        s += "-"*ml
        return s

# DONEs
class LinearLayer:

    # DONE 
    def __init__(self,
                 n_cels_in,
                 n_cels_out, 
                 bias = True, 
                 activation = 'ReLU',
                 batch = 1,
                 random_seed = None):
        """
        Parameters
        ----------
            · n_cels_in:
            · n_cels_out:
            · bias:
            · activation: defines wich activation layer we want to apply at the output.
        """
        if random_seed != None:
            np.random.seed(random_seed)

        if activation not in NeuralNetwork.ActivationFunctions:
            raise TypeError(f"Please, use some of the \033[1;31mavailable activation funcions: {NeuralNetwork.ActivationFunctions}\033[0m")
        
        self.n_in = n_cels_in
        self.n_out = n_cels_out
        self.n_trainable_parameters = self.n_out*self.n_in + (self.n_out if bias else 0)
        
        # Initialize weights randomly and normalize
        limit = np.sqrt(1 / self.n_in)
        self.weights = np.random.randn(self.n_in, self.n_out) * limit
        self.bias = np.random.randn(self.n_out) * limit

        self.activation = NeuralNetwork.AF[activation]
        
        # Values of the neurons
        self.x = np.zeros(shape=(self.n_in,))
        self.z = np.zeros(shape=(self.n_out,))
        self.cache = np.zeros(shape=(self.n_out, batch)) # The error of a Neuron
    
    # DONE 
    def forward(self, input):
        # Update values of the neurons
        self.x = input
        self.z = input @ self.weights + self.bias 
        return self.activation(self.z)
        
    
    # DONE
    def backpropagation(self, d, first_delta_computed = False):
        
        if first_delta_computed:
            self.cache = np.array(d)
        else:
            self.cache =  d * self.activation.partial(self.z)
        return self.cache @ self.weights.T

    # DONE TODO - test
    def update_parameters(self, learning_rate, batch = 1, adam = True):
        # Verify if we are working by batches
        #breakpoint()
        self.weights = self.weights - learning_rate*np.outer(self.x, self.cache)  
        # PROBLEM WITH DIMENSIONALITY: REVISAR FÓRMULAS 
        if batch == 1:
            self.bias = self.bias - (learning_rate/batch)* self.cache
        else:
            self.bias = self.bias - (learning_rate/batch)* np.sum(self.cache, axis=0)
    
    # DONE
    def __str__(self):
        s = f"dim(\033[1;33m{self.n_in}\033[0m) -- Fully Conected --> dim(\033[1;33m{self.n_out}\033[0m) --> \033[1;32m{str(self.activation)}\033[0m"
        return s

# DONE
class Sequence:

    # DONE
    def __init__(self, layers):
        # Verify that the dimensions are correct
        n = len(layers)
        for i in range(n-1):
            if layers[i].n_out != layers[i+1].n_in:
                raise ValueError("Incompatible layer dimensions")
        
        self.layers = layers
        self.n_layers = n
        self.size_in = layers[0].n_in if n != 0 else 0
        self.size_out = layers[-1].n_out if n != 0 else 0

    # DONE
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input

    # DONE 
    def add(self, layer):
        self.layers.append(layer)
    
    def __getitem__(self,index):
        return self.layers[index]
    
    def __len__(self):
        return len(self.layers)
    
    # DONE
    def backpropagate(self, delta, step=0.1, update_parameters = False, first_delta_computed = False):
        #breakpoint()
        
        n = len(self.layers)
        for i in range(n):
            layer = self.layers[n-1-i]
            if i == 0:
                delta = layer.backpropagation(delta, first_delta_computed = True)
            else:
                delta = layer.backpropagation(delta)

            if update_parameters:
                layer.update_parameters(learning_rate = step)
        return delta
            

        
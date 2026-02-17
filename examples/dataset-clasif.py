from nn.NN import *
import pandas as pd

"""
This script compares the 4 following algorithms to train a simple Neural Network:
    1. GradDesc. with no modifications: for each forward run, we backpropagate and update parameters
    2. GradDesc. ADAM powered: for each forward run, we backpropagate using ADAM algorithm
    3. StochasticGradDesc with no modifications.
    4. StochasticGradDesc ADAM powered.
"""
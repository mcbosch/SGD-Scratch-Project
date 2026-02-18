import numpy as np
import pandas as pd

from nn.NN import *
from nn.utils import *

# Load data
toy_data = pd.read_csv("examples\datasets\square_regions.csv")
# Build model
model = NeuralNetwork([2,4,2],activations=["ReLU","Softmax"])
#df_train, df_test = split(toy_data, 0.8)
x_train, y_train = list(zip(list(toy_data["x"]),list(toy_data["y"]))), toy_data["label"]

data = list(zip(x_train,y_train))
model.train(data,50,0.1)
acc = model.test(data)
print(acc)
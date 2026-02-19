import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nn.NN import *
from nn.utils import *


# Load data
toy_data = pd.read_csv("examples\datasets\square_regions.csv")
# Build model
model = NeuralNetwork([2,4,2],activations=["ReLU","Softmax"])
#df_train, df_test = split(toy_data, 0.8)
x_train, y_train = list(zip(list(toy_data["x"]),list(toy_data["y"]))), toy_data["label"]

epochs = 50
data = list(zip(x_train,y_train))
loss_ep, t_ep = model.train(data,epochs,0.1)

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color=color)
ax1.plot(list(range(epochs)), loss_ep, color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('time', color=color)
ax2.plot(list(range(epochs)), t_ep, color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
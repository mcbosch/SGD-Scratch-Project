from nn.NN import *
import numpy as np
import matplotlib.pyplot as plt

data = np.load("mnist-test\\mnist.npz")

# Print Data shape:
x_train = data['x_train']
y_train = data['y_train']

print("Datapoints: ", len(x_train))
print("Shape: ", x_train[0].shape)


# Transforming to vectors
print("Transforming matrixs to vectors of shape: ", x_train[0].shape[0]*x_train[0].shape[1])
x_train = [x_train[i].reshape(-1) for i in range(len(x_train))]

model = NeuralNetwork([784,125,125,10],activations=["ReLU","ReLU","Softmax"])

epochs = 100
data = list(zip(x_train,y_train))
loss_ep, t_ep = model.train(data,epochs,0.01,loss=MSE())

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

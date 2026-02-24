from nn.NN import *
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load("mnist-test\\mnist.npz")

# Print Data shape:
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
# Transforming to vectors
x_train = [x_train[i].reshape(-1)*1/255 for i in range(len(x_train))] # Normalize data to avoid overflow problems 
model = NeuralNetwork([LinearLayer(784,524,activation="ReLU"),
                       LinearLayer(524,124,activation="ReLU"),
                       LinearLayer(124,10,activation="Softmax")])

x_test = [x_test[i].reshape(-1)*1/255 for i in range(len(x_test))]
data_test = list(zip(x_test, y_test))

epochs = 100
data = list(zip(x_train,y_train))
data_train, data_val = split(data, 0.8)
n_classes = len(np.unique(y_train))

print(f"\nDataset: \tTrain: {len(data_train)} | Val: {len(data_val)} | Test: {len(data_test)} | Clases: {np.unique(y_train)}\n")

results = model.train(data_train,epochs,0.01,batch_size=250,loss=CrossEntropy(),adam=True, data_val = data_val)


fig, axes = plt.subplots(1,3,figsize=(15, 4))
epochs_range = list(range(epochs))

# Plot loss
axes[0].plot(epochs_range, results['Epochs']['loss_train'], label='Train Loss', color='steelblue', linewidth=2)
axes[0].plot(epochs_range, results['Epochs']['loss_val'],   label='Val Loss',   color='tomato',    linewidth=2, linestyle = '--')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Loss training and validation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot acc val
axes[1].plot(epochs_range,[a*100 for a in results['Epochs']['acc_val']], label='Accuracy Validation', color='purple', linewidth=2)
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('accuracy (%)')
axes[1].set_ylim(0,105)

axes[1].axhline(y=100, color = 'red', alpha=0.5, linewidth = 2, linestyle = '--', label = 'Perfect Acc' )
axes[1].axhline(y=100/n_classes, color = 'gray', alpha = 0.5, linewidth = 2, linestyle = '--', label = 'Baseline, random predictions' )

axes[1].set_title('Accuracy Validation per Epoch')
axes[1].legend()
axes[1].grid(True, alpha = 0.3)

# Results table
acc_test, ls_test = model.test(data_test, CrossEntropy())

axes[2].table(cellText= [[int(acc_test*100)], [round(ls_test,3)], [round(results['Time']/60,5)], [len(data_train)], [len(data_test)]],

              cellColours = [['limegreen'],['lightcoral'],['aquamarine'], ['gray'], ['gray']],
              cellLoc = 'center',
              colLoc = 'center',
              rowLoc = 'center',
              rowLabels = ['Accuracy Test', 'Loss Test', 'Time Training (min)', 'Num Datapoints Train', 'Num Datapoints Test'],
              loc = 'center')
axes[2].axis('off')
axes[2].axis('tight')
fig.suptitle(f"NN (784 -> 524 -> 124 -> 10)")
plt.tight_layout()
plt.savefig(f'mnist-test\\img\\results-mnist.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot with results saved in mnist-test\\img\\results-mnist.png")
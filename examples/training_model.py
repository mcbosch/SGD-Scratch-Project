import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nn.NN import *
from nn.utils import *
"""
Why Loss Now increases with random batches?
"""

# Load data
dataset = {'square': "examples\datasets\square_regions.csv",
           'circle': "examples\datasets\circle_regions.csv"}
data_name = 'square'
toy_data = pd.read_csv(dataset[data_name])

#df_train, df_test = split(toy_data, 0.8)
X, Y = list(zip(list(toy_data["x"]),list(toy_data["y"]))), toy_data["label"]
data = list(zip(X,Y))
data_train, data_test = split(data, 0.8)
data_train, data_val = split(data_train, 0.8)

model = NeuralNetwork([LinearLayer(2,3,activation="Softmax"),
                       LinearLayer(3,2,activation="Softmax")])


n_classes = len(np.unique(Y))
epochs = 100
print(f"\nDataset: \tTrain: {len(data_train)} | Val: {len(data_val)} | Test: {len(data_test)} | Clases: {list(range(n_classes))}\n")

results = model.train(data_train,epochs,0.01,batch_size=20,loss=CrossEntropy(),adam=True, data_val = data_val)


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

axes[2].table(cellText= [[int(acc_test*100)], [round(ls_test,3)], [round(results['Time'],5)], [len(data_train)], [len(data_test)]],

              cellColours = [['limegreen'],['lightcoral'],['aquamarine'], ['gray'], ['gray']],
              cellLoc = 'center',
              colLoc = 'center',
              rowLoc = 'center',
              rowLabels = ['Accuracy Test', 'Loss Test', 'Time Training', 'Num Datapoints Train', 'Num Datapoints Test'],
              loc = 'center')
axes[2].axis('off')
axes[2].axis('tight')

plt.tight_layout()
plt.savefig(f'examples\\results-{data_name}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot with results saved in examples\\results-{data_name}.png")
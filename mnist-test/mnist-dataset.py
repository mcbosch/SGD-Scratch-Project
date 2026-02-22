from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

# Si tu VAE es numpy puro y necesitas arrays:
X_train = train_data.data.numpy() / 255.0  # (60000, 28, 28)
X_test  = test_data.data.numpy()  / 255.0

# Aplanar si tu encoder espera vectores
X_train = train_data.data.numpy() / 255.0
y_train = train_data.targets.numpy()  # etiquetas 0-9

X_test = test_data.data.numpy() / 255.0
y_test = test_data.targets.numpy()

X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

# Split train/val
split = int(0.8 * len(X_train))
X_val, y_val     = X_train[split:], y_train[split:]
X_train, y_train = X_train[:split], y_train[:split]

breakpoint()
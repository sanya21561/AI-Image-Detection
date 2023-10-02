import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
data_dir = 'data'
dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# PyTorch se Logistic Regression

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return torch.sigmoid(self.fc(x))

input_dim = 32 * 32 * 3
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.float().view(-1, 1)).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy (PyTorch): {accuracy:.2f}%')


#Numpy mein karke scikit learn se

def flatten_tensor(tensor):
    return tensor.view(tensor.size(0), -1).numpy()

X_train = np.vstack([flatten_tensor(inputs) for inputs, _ in train_loader])
y_train = np.hstack([labels.numpy() for _, labels in train_loader])

X_val = np.vstack([flatten_tensor(inputs) for inputs, _ in val_loader])
y_val = np.hstack([labels.numpy() for _, labels in val_loader])

sklearn_lr = LogisticRegression()
sklearn_lr.fit(X_train, y_train)

y_pred = sklearn_lr.predict(X_val)
accuracy_sklearn = accuracy_score(y_val, y_pred) * 100
print(f'Validation Accuracy (scikit-learn): {accuracy_sklearn:.2f}%')

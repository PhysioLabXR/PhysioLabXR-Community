import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int32)

    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

data = pickle.load(open('../../../image_data.p', 'rb'))
label = pickle.load(open('../../../image_labels.p', 'rb'))
# for i in range(len(data)):
#     plt.imshow(data[i])
#     plt.title(f'Class: {label[i]}')
#     plt.show()

data = np.transpose(np.stack(data), (0, 3, 1, 2))
label_index = np.unique(np.array(label, dtype=int))


# Load the VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Freeze the weights of the convolutional layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one
vgg16.classifier[-1] = nn.Linear(4096, len(label_index))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters())



# Create DataLoaders for train, validation, and test sets
batch_size = 64
shuffle = True  # Set to True if you want to shuffle the data

# Load the dataset
image_dataset = ImageDataset(data, label)

# Define the sizes for train, validation, and test sets
train_size = int(0.7 * len(image_dataset))  # 70% of the data for training
test_size = len(image_dataset) - train_size  # Remaining data for testing

# Use random_split to create train, validation, and test datasets
train_dataset, test_dataset = random_split(image_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Train the model
for epoch in range(200):
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    vgg16.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels - 1
        optimizer.zero_grad()

        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss = criterion(outputs, torch.tensor(labels, dtype=torch.long))
        l2_reg = torch.tensor(0.)
        for param in vgg16.parameters():
            l2_reg += torch.norm(param)

        loss += 0.001 * l2_reg  # Adjust the regularization strength

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        print('[%d, %5d] train accuracy: %.3f' % (epoch + 1, i + 1, train_correct / train_total))
        running_loss = 0.0

    vgg16.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels - 1
            outputs = vgg16(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %.3f' % (test_correct / test_total))




print(len(data))
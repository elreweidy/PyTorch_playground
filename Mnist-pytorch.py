import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST("", train=False, download=True, transform=transforms.ToTensor())

train_set = DataLoader(train, batch_size=64, shuffle=True)
test_set = DataLoader(test, batch_size=64, shuffle=True)


# SOME PREPROCESSING

# for data in train_set:
#     print(data)
#     break
# X, y = data[0][0], data[1][0]
# plt.imshow(data[0][9].view(28, 28))
# plt.show()
#
# total = 0
# counter_dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
# for data in train_set:
#     Xs, ys = data
#     for y in ys:
#         counter_dic[int(y)] += 1
#         total += 1
# print(counter_dic, total)
#
# for i in counter_dic:
#     print(f'{i}: {counter_dic[i] / total * 100}')
#
# ---------------------------------------
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# ----------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


# HYPER-PARAMETERS
lr = 0.001
epochs = 5

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_set):
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy in the training data")
    else:
        print("checking accuracy in the test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():

        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} out of {num_samples}: {float(num_correct) / float(num_samples) * 100:.2f}%")
    model.train()


check_accuracy(train_set, model)
check_accuracy(test_set, model)

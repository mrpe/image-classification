import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()  # Inherit methods from the super class which this class extends from
        # 3 because color image / red green blue, 6 feature maps, 5 size of the kernels (matrix)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(71148, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 10)
        self.fc4 = nn.Linear(10, 2)

        self.relu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, sample):
        x = self.max_pool(self.relu(self.conv1(sample)))
        x = self.max_pool(self.relu(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def __call__(self, sample):
        return self.forward(sample)

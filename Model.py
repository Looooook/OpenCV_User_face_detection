import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 15, kernel_size=(25, 25))
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding='same')  # 32 100 100

        # self.mp1 = torch.nn.MaxPool2d(kernel_size=(4, 4))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 32 50 50

        # self.conv2 = torch.nn.Conv2d(15, 30, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)  # 64 50 50

        # self.mp2 = torch.nn.MaxPool2d(kernel_size=(3, 3))
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 64 25 25

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(2, 2))  # 128 24 24

        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 128 12 12

        # self.linear = torch.nn.Linear(750, 4)
        self.linear1 = torch.nn.Linear(18432, 3072)
        self.linear2 = torch.nn.Linear(3072, 512)
        # self.linear3 = torch.nn.Linear(5000, 1000)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(128, 2)

    def forward(self, x):
        in_size = x.size(0)
        # print(x.shape)
        # print(x.size)
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = self.mp3(x)
        # print(x.shape)
        x = x.view(in_size, -1)
        # print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        return torch.sigmoid(x)



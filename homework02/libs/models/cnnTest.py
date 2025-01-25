import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class CNNTest:

    operation = []
    batch_size = 0
    learning_rate = 0.0
    images = None
    labels = None
    transforms = None

    def __init__(self, batch_size=64, learning_rate=0.001, operation=[]):
        self.operation = operation
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Combine transformations and normalization
        self.transform = transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                                        download=True, transform=self.transform)

        self.test_dataset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                                        download=True, transform=self.transform)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                        shuffle=False)

        self.dataiter = iter(self.train_loader)
        self.images, self.labels = self.dataiter.__next__()

    def calculateFc1(self):
        x = self.images
        print(x.shape)
        for op in self.operation:
            x = op(x)
            print(x.shape)
        return x


if __name__ == '__main__':
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    relu = nn.ReLU()
    dropout = nn.Dropout(0.25)

    bn1 = nn.BatchNorm2d(32)
    bn2 = nn.BatchNorm2d(64)
    bn3 = nn.BatchNorm2d(128)

    conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    cnnTest = CNNTest(operation=[conv1, relu, bn1, pool, conv2, relu, bn2, pool, conv3, relu, bn2, conv4, relu, bn3, conv5, bn3, relu])
    x = cnnTest.calculateFc1()
    print("fc1(last three element):", x.shape)
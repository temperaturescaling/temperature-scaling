import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_class=10, activation='ReLU'):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 256)
        self.fc2 = nn.Linear(256, n_class)
        self.activation = getattr(nn, activation)()
        self.max_pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.activation(self.max_pool(self.conv1(x)))
        x = self.activation(self.max_pool(self.conv2(x)))
        x = x.view(-1, 500)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def penultimate_feature(self, x):
        x = self.activation(self.max_pool(self.conv1(x)))
        x = self.activation(self.max_pool(self.conv2(x)))
        x = x.view(-1, 500)
        x = self.fc1(x)
        return x
    
if __name__ == '__main__':
    model = CNN(n_class=10)
    x = torch.randn((2,3,32,32))
    o = model(x)
    print(o.shape)
    
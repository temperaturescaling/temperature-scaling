import torch
import torch.nn as nn
import torchvision.models as models

def ResNet18(n_class=100):
    net = models.resnet18()
    net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)
    return net

# class ResNet18(nn.Module):
#     def __init__(self):
#         super(ResNet18, self).__init__()
#         self = get_resnet('resnet18')



if __name__ == '__main__':
    # net = get_resnet('resnet18')
    net = ResNet18()
    X = torch.rand((2,3,32,32))
    pred = net(X)
    print(net)
    print(pred.shape)
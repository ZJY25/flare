import torch
import torch.nn as nn
from thop import clever_format
from utils import network_parameters


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv1x1(3, 3)

    def forward(self, x):
        y = self.conv1(x)
        return x, y


if __name__ == '__main__':
    input = torch.randn(2, 3, 225, 225)
    model = Net()
    out = model(input)
    # print(model)
    p_number = network_parameters(model)
    p_number = clever_format([p_number], "%.3f")
    print(">>>> model Param.: ", p_number)
    print(out.shape)
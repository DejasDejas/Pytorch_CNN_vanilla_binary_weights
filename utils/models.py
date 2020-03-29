import torch.nn as nn
import torch.nn.functional as F

from utils.activations import DeterministicBinaryActivation, StochasticBinaryActivation
from utils.functions import Hardsigmoid


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.mode = None
        self.estimator = None


class NonBinaryNet(Net):

    def __init__(self):
        super(NonBinaryNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2))
        self.act = Hardsigmoid()
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, input):
        x, slope = input
        x_layer1 = self.act(self.layer1(x) * slope)
        x_layer2 = self.act(self.layer2(x_layer1))
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)
        x_fc = self.fc(x_layer2)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out


class BinaryNet(Net):

    def __init__(self, first_conv_layer, last_conv_layer, mode='Deterministic', estimator='ST'):
        super(BinaryNet, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        # if mode == 'Deterministic':
        #    assert estimator == 'ST'

        self.mode = mode
        self.estimator = estimator
        self.first_conv_layer = first_conv_layer
        self.last_conv_layer = last_conv_layer

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2))
        if self.first_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer1 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer1 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer1 = Hardsigmoid()
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2))
        if self.last_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer2 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer2 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer2 = Hardsigmoid()
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, input):
        x, slope = input
        if self.first_conv_layer:
            x_layer1 = self.act_layer1((self.layer1(x), slope))
        else:
            x_layer1 = self.act_layer1(self.layer1(x) * slope)
        if self.last_conv_layer:
            x_layer2 = self.act_layer2((self.layer2(x_layer1), slope))
        else:
            x_layer2 = self.act_layer2(self.layer2(x_layer1) * slope)
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)
        x_fc = self.fc(x_layer2)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out

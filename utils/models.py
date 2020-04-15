import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.activations import DeterministicBinaryActivation, StochasticBinaryActivation
from utils.functions import Hardsigmoid


# loading the saved model
def fetch_last_checkpoint_model_filename(model_save_path):
    import os
    checkpoint_files = os.listdir(model_save_path)
    checkpoint_files = [f for f in checkpoint_files if '.pth' in f]
    checkpoint_iter = [
        int(x.split('_')[-1].split('.')[0])
        for x in checkpoint_files]
    last_idx = np.array(checkpoint_iter).argmax()
    return os.path.join(model_save_path, checkpoint_files[last_idx])
    
    
# Model, activation type, estimator type
def get_my_model_MNIST(binary, stochastic=True, reinforce=False, first_conv_layer=True,
                 last_conv_layer=False):
                     
    # Slope annealing
    if binary:
        def get_slope(number_epoch): return 1.0 * (1.005 ** (number_epoch - 1))
    else:
        def get_slope(number_epoch): return 1.0

    if binary:
        if stochastic:
            mode = 'Stochastic'
            names_model = 'Stochastic'
        else:
          
            mode = 'Deterministic'
            names_model = 'Deterministic'
        if reinforce:
            estimator = 'REINFORCE'
            names_model += '_REINFORCE'
        else:
            estimator = 'ST'
            names_model += '_ST'
        if first_conv_layer:
            names_model += '_first_conv_binary'
        if last_conv_layer:
            names_model += '_last_conv_binary'
        model = BinaryNetMNIST(first_conv_layer=first_conv_layer, 
                          last_conv_layer=last_conv_layer, mode=mode, 
                          estimator=estimator)
    else:
        model = NoBinaryNetMnist()
        names_model = 'NonBinaryNet'
        mode = None
        estimator = None

    # gpu config:
    # model, use_gpu = gpu_config(model)
    return model, names_model
    

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.mode = None
        self.estimator = None


class NoBinaryNetMnist(Net):

    def __init__(self):
        super(NoBinaryNetMnist, self).__init__()

        self.layer1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_layer1 = Hardsigmoid()
        self.layer2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_layer2 = Hardsigmoid()
        self.fc = nn.Linear(7*7*20, 10)



    def forward(self, input):
        x = input
        slope = 1.0
        x_layer1 = self.act_layer1(self.maxpool1(self.batchnorm1(self.layer1(x) * slope)))
        x_layer2 = self.act_layer2(self.maxpool2(self.batchnorm2(self.layer2(x_layer1) * slope)))
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)
        x_fc = self.fc(x_layer2)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out


class BinaryNetMNIST(Net):

    def __init__(self, first_conv_layer, last_conv_layer, mode='Deterministic', estimator='ST'):
        super(BinaryNetMNIST, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        # if mode == 'Deterministic':
        #    assert estimator == 'ST'

        self.mode = mode
        self.estimator = estimator
        self.first_conv_layer = first_conv_layer
        self.last_conv_layer = last_conv_layer

        self.layer1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.first_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer1 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer1 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer1 = Hardsigmoid()
        self.layer2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.last_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer2 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer2 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer2 = Hardsigmoid()
        self.fc = nn.Linear(7*7*20, 10)

    def forward(self, input):
        x = input
        slope = 1.0
        if self.first_conv_layer:
            x_layer1 = self.act_layer1(((self.maxpool1(self.batchnorm1(self.layer1(x)))), slope))
        else:
            x_layer1 = self.act_layer1(self.maxpool1(self.batchnorm1(self.layer1(x) * slope)))
        if self.last_conv_layer:
            x_layer2 = self.act_layer2(((self.maxpool2(self.batchnorm2(self.layer2(x_layer1)))), slope))
        else:
            x_layer2 = self.act_layer2(self.maxpool2(self.batchnorm2(self.layer2(x_layer1) * slope)))
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)

        x_fc = self.fc(x_layer2)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out
        

class NoBinaryNetOmniglotClassification(Net):

    def __init__(self):
        super(NoBinaryNetOmniglotClassification, self).__init__()

        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_layer1 = nn.ReLU()
        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)            
        self.act_layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.act_layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)              
        self.act_layer4 = nn.ReLU()
        self.fc = nn.Linear(6 * 6 * 64, 1623)

    def forward(self, input):
        if len(input)==2:
            x, slope = input
        else:
            x = input
            slope = 1.0
        x_layer1 = self.act_layer1(self.maxPool1(self.batchNorm1(self.layer1(x) * slope)))
        x_layer2 = self.act_layer2(self.maxPool2(self.batchNorm2(self.layer2(x_layer1))))
        x_layer3 = self.act_layer3(self.maxPool3(self.batchNorm3(self.layer3(x_layer2))))
        x_layer4 = self.act_layer4(self.maxPool4(self.batchNorm4(self.layer4(x_layer3))))
        x_layer4 = x_layer4.view(x_layer4.size(0), -1)
        x_fc = self.fc(x_layer4)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out


class BinaryNetOmniglotClassification(Net):

    def __init__(self, first_conv_layer, second_conv_layer, third_conv_layer, fourth_conv_layer, mode='Deterministic', estimator='ST'):
        super(BinaryNetOmniglotClassification, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        # if mode == 'Deterministic':
        #    assert estimator == 'ST'

        self.mode = mode
        self.estimator = estimator
        self.first_conv_layer = first_conv_layer
        self.second_conv_layer = second_conv_layer
        self.third_conv_layer = third_conv_layer
        self.fourth_conv_layer = fourth_conv_layer

        self.layer1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.first_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer1 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer1 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer1 = nn.ReLU()
        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.second_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer2 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer2 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2) 
        if self.third_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer3 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer3 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)   
        if self.fourth_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer4 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer4 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer4 = nn.ReLU()
        self.fc = nn.Linear(6 * 6 * 64, 1623)

    def forward(self, input):
        if len(input)==2:
            x, slope = input
        else:
            x = input
            slope = 1.0
        if self.first_conv_layer:
            x_layer1 = self.act_layer1(((self.maxPool1(self.batchNorm1(self.layer1(x)))), slope))
        else:
            x_layer1 = self.act_layer1(self.maxPool1(self.batchNorm1(self.layer1(x) * slope)))
        if self.second_conv_layer:
            x_layer2 = self.act_layer2(((self.maxPool2(self.batchNorm2(self.layer2(x_layer1)))), slope))
        else:
            x_layer2 = self.act_layer2(self.maxPool2(self.batchNorm2(self.layer2(x_layer1) * slope)))
        if self.third_conv_layer:
            x_layer3 = self.act_layer3(((self.maxPool3(self.batchNorm3(self.layer3(x_layer2)))), slope))
        else:
            x_layer3 = self.act_layer3(self.maxPool3(self.batchNorm3(self.layer3(x_layer2) * slope)))
        if self.fourth_conv_layer:
            x_layer4 = self.act_layer4(((self.maxPool4(self.batchNorm4(self.layer4(x_layer3)))), slope))
        else:
            x_layer4 = self.act_layer4(self.maxPool4(self.batchNorm4(self.layer4(x_layer3) * slope)))
        x_layer4 = x_layer4.view(x_layer4.size(0), -1)
        x_fc = self.fc(x_layer4)
        x_out = F.log_softmax(x_fc, dim=1)
        return x_out


class NoBinaryMatchingNetwork(nn.Module):
    def __init__(self, n: int, k: int, q: int, num_input_channels: int):
        """Creates a Matching Network as described in Vinyals et al.
        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(NoBinaryMatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)

    def forward(self, inputs):
        pass


def get_few_shot_encoder(num_input_channels=1) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].
    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class BinaryMatchingNetwork(nn.Module):
    def __init__(self, first_conv_layer, second_conv_layer, third_conv_layer, fourth_conv_layer,
                 n: int, k: int, q: int, num_input_channels: int, mode='Deterministic', estimator='ST'):

        super(BinaryMatchingNetwork, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']

        self.mode = mode
        self.estimator = estimator
        self.first_conv_layer = first_conv_layer
        self.second_conv_layer = second_conv_layer
        self.third_conv_layer = third_conv_layer
        self.fourth_conv_layer = fourth_conv_layer
        self.n = n
        self.k = k
        self.q = q
        self.num_input_channels = num_input_channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if first_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer1 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer1 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer1 = nn.ReLU()
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if second_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer2 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer2 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer2 = nn.ReLU()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if third_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer3 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer3 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer3 = nn.ReLU()
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if fourth_conv_layer:
            if self.mode == 'Deterministic':
                self.act_layer4 = DeterministicBinaryActivation(estimator=estimator)
            elif self.mode == 'Stochastic':
                self.act_layer4 = StochasticBinaryActivation(estimator=estimator)
        else:
            self.act_layer4 = nn.ReLU()

    def forward(self, inputs):
        x, slope = inputs
        if self.first_conv_layer:
            x_layer1 = self.act_layer1((self.layer1(x), slope))
        else:
            x_layer1 = self.act_layer1(self.layer1(x) * slope)
        if self.second_conv_layer:
            x_layer2 = self.act_layer2((self.layer2(x_layer1), slope))
        else:
            x_layer2 = self.act_layer2(self.layer2(x_layer1) * slope)
        if self.third_conv_layer:
            x_layer3 = self.act_layer3((self.layer3(x_layer2), slope))
        else:
            x_layer3 = self.act_layer3(self.layer3(x_layer2) * slope)
        if self.fourth_conv_layer:
            x_layer4 = self.act_layer4((self.layer4(x_layer3), slope))
        else:
            x_layer4 = self.act_layer4(self.layer4(x_layer3) * slope)
        x_out = x_layer4.view(x_layer4.size(0), -1)
        return x_out

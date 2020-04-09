import torch.optim as optim
from DataLoader.dataLoaders import get_mnist_dataloaders
from utils.training import training, test, gpu_config
from torch import load
import torch

from utils.models import NoBinaryNetMnist, BinaryNet

"""
Code source: https://github.com/Wizaron/binary-stochastic-neurons
"""

# parameters default values
lr = 0.1
momentum = 0.9
nb_epoch = 10
batch_size_train = 64
batch_size_test = 1000
slope_annealing = False
reinforce = False
stochastic = False
binary = False
plot_result = False
first_conv_layer = False
last_conv_layer = False
omniglot = True

# Model, activation type, estimator type
if binary:
    if stochastic:
        mode = 'Stochastic'
        names_model = 'MNIST/Stochastic'
    else:
        mode = 'Deterministic'
        names_model = 'MNIST/Deterministic'
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
    model = BinaryNet(first_conv_layer=first_conv_layer, last_conv_layer=last_conv_layer,
                      mode=mode, estimator=estimator)
else:
    model = NoBinaryNetMnist()
    names_model = 'MNIST/NonBinaryNet'
    mode = None
    estimator = None

# gpu config:
model, use_gpu = gpu_config(model)

# MNIST Dataset
train_loader, valid_loader, test_loader, classes = get_mnist_dataloaders(batch_size_train, batch_size_test)


# visualize example
# show_som_examples(train_loader)

# train
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# train_loss, train_acc, val_loss, val_acc = training(use_gpu, model, names_model, nb_epoch, train_loader, test_loader,
#                                                    optimizer, plot_result, slope_annealing)
# test
model.load_state_dict(load('./trained_models/' + names_model + '.pt', map_location=torch.device('cpu')))
test_loss, test_acc = test(use_gpu, model, test_loader)

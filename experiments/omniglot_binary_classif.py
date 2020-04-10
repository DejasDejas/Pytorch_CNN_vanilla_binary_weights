import torch.optim as optim
from DataLoader.dataLoaders import get_omniglot_dataloaders_v1
from utils.models import NoBinaryNetOmniglotClassification, BinaryNetOmniglotClassification
from utils.training import gpu_config
from utils.callback import *
from visualize.viz import show_som_examples
from utils.training import training
from config import PATH

# parameters default values
lr = 1e-3
# momentum = 0.9
nb_epoch = 50
batch_size_train = 64
batch_size_test = 128

slope_annealing = True
reinforce = False
stochastic = True
binary = True
plot_result = False
first_conv_layer = True
second_conv_layer = False
third_conv_layer = False
fourth_conv_layer = False
omniglot = True


# Model, activation type, estimator type
if binary:
    if stochastic:
        mode = 'Stochastic'
        names_model = 'Omniglot_classif_Stochastic'
    else:
        mode = 'Deterministic'
        names_model = 'Omniglot_classif_Deterministic'
    if reinforce:
        estimator = 'REINFORCE'
        names_model += '_REINFORCE'
    else:
        estimator = 'ST'
        names_model += '_ST'
    if first_conv_layer:
        names_model += '_first_conv_binary'
    if second_conv_layer:
        names_model += '_second_conv_binary'
    if third_conv_layer:
        names_model += '_third_conv_binary'
    if fourth_conv_layer:
        names_model += '_fourth_conv_binary'
    model = BinaryNetOmniglotClassification(first_conv_layer=first_conv_layer, second_conv_layer=second_conv_layer,
                                            third_conv_layer=third_conv_layer, fourth_conv_layer=fourth_conv_layer,
                                            mode=mode, estimator=estimator)
else:
    model = NoBinaryNetOmniglotClassification()
    names_model = 'Omniglot_classif_NonBinaryNet'
    mode = None
    estimator = None

# gpu config:
model, use_gpu = gpu_config(model)

# Omniglto Dataset
train_loader, test_loader = get_omniglot_dataloaders_v1(batch_size_train, batch_size_test)

# visualize example
# show_som_examples(train_loader)

path_save_plot = '/results/Omniglot_classif/plot_acc_loss/'
path_save_model = '/trained_models/Omniglot_classif/'

# train
"""
callbacks = [
    ModelCheckpoint(
        filepath=PATH + path_save_model + names_model + '.pth',
        monitor=f'val_classif_{names_model}acc',
    ),
    ReduceLROnPlateau(patience=20, factor=0.5, monitor=f'val_classif_{names_model}_acc'),
    CSVLogger(PATH + f'/logs/omniglot_classif/{names_model}.csv'),
]
"""
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
train_loss, train_acc, val_loss, val_acc = training(path_save_plot, path_save_model, use_gpu, model, names_model,
                                                    nb_epoch, train_loader, test_loader,
                                                    optimizer, plot_result, slope_annealing)

# test
# model.load_state_dict(load('./trained_models/' + names_model + '.pt', map_location=torch.device('cpu')))
# test_loss, test_acc = test(use_gpu, model, test_loader)

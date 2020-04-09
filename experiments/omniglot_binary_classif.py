import torch.optim as optim
from DataLoader.dataLoaders import get_omniglot_dataloaders_v1
from utils.models import NoBinaryNetOmniglotClassification, BinaryNetOmniglotClassif
from utils.training import gpu_config
from visualize.viz import show_som_examples
from utils.training import training
from torchsummary import summary


# parameters default values
lr = 0.1
momentum = 0.9
nb_epoch = 10
batch_size_train = 32
batch_size_test = 32
slope_annealing = True
reinforce = False
stochastic = True
binary = True
plot_result = False
first_conv_layer = True
second_conv_layer = False
third_conv_layer = False
omniglot = True


# Model, activation type, estimator type
if binary:
    if stochastic:
        mode = 'Stochastic'
        names_model = 'Omniglot_classif/Stochastic'
    else:
        mode = 'Deterministic'
        names_model = 'Omniglot_classif/Deterministic'
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
    model = BinaryNetOmniglotClassif(first_conv_layer=first_conv_layer, second_conv_layer=second_conv_layer,
                                     third_conv_layer=third_conv_layer, mode=mode, estimator=estimator)
else:
    model = NoBinaryNetOmniglotClassification()
    names_model = 'Omniglot_classif/NonBinaryNet'
    mode = None
    estimator = None

# gpu config:
model, use_gpu = gpu_config(model)

# Omniglto Dataset
train_loader, test_loader = get_omniglot_dataloaders_v1(batch_size_train, batch_size_test)

# visualize example
# show_som_examples(train_loader)

# train
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
train_loss, train_acc, val_loss, val_acc = training(use_gpu, model, names_model, nb_epoch, train_loader, test_loader,
                                                    optimizer, plot_result, slope_annealing)

# test
# model.load_state_dict(load('./trained_models/' + names_model + '.pt', map_location=torch.device('cpu')))
# test_loss, test_acc = test(use_gpu, model, test_loader)


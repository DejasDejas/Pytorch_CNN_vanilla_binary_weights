import sys
sys.path.append('')
import torch.optim as optim
from DataLoader.dataLoaders import get_mnist_dataloaders
from utils.training import run
from utils.models import get_my_model_MNIST
import torch.nn.functional as F


# parameters default values
lr = 0.1
momentum = 0.9
nb_epoch = 10
criterion = F.nll_loss
batch_size_train = 64
batch_size_test = 1000
omniglot = True


# gpu config:
model, name_model = get_my_model_MNIST(binary=False)

# MNIST Dataset
train_loader, valid_loader, test_loader, classes = get_mnist_dataloaders(batch_size_train, batch_size_test)

# visualize example
# show_som_examples(train_loader)

# train
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
path_model_checkpoint = ''
path_save_plot = ''
log_interval = 2
run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, nb_epoch, lr, momentum,
    criterion, log_interval)

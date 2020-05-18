import sys
sys.path.append('')
import torch.nn.functional as F
from utils.models import get_my_model_Omniglot
from DataLoader.dataLoaders import get_omniglot_dataloaders_classification
from utils.training import run

batch_size_train = 64
batch_size_test = 64
# Dataset
train_loader, valid_loader, test_loader = get_omniglot_dataloaders_classification(batch_size_train, batch_size_test)

# parameters default values
epochs = 50
lr = 1e-3
momentum = 0.9
log_interval = 10  # how many batches to wait before logging training status
criterion = F.nll_loss

# parameters model to load no Binary model with stride
binary = False
maxpooling = False
model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/No_binary_models/stride/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

# parameters model to load no Binary model with maxpooling
binary = False
maxpooling = True
model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/No_binary_models/maxpooling/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

# parameters model to load Binary model stride
binary = True
maxpooling = False
model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/Binary_models/stride/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

# parameters model to load Binary model maxpooling
binary = True
maxpooling = True
model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/Binary_models/maxpooling/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

# parameters model to load mixt model stride
binary = False
maxpooling = False
mixt=True

model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling, mixt=mixt)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/Mixt_models/stride/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

# parameters model to load mixt model maxpooling
binary = False
maxpooling = True
mixt=True

model, name_model = get_my_model_Omniglot(binary, maxpooling=maxpooling, mixt=mixt)
print(name_model)

path_model_checkpoint = 'trained_models/Omniglot_classif/Mixt_models/maxpooling/'
path_save_plot = 'results/Omniglot_results/plot_acc_loss/Omniglot_classif/'

run(model, path_model_checkpoint, path_save_plot, name_model, train_loader, valid_loader, epochs, lr, momentum, criterion, log_interval)

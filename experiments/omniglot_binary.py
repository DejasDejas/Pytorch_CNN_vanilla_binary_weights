"""
Reproduce Matching Network results of Vinyals et al
code source:
"""
from torch.optim import Adam

from DataLoader.dataset import OmniglotDataset
from utils.core import prepare_nshot_task, EvaluateFewShot
from utils.models import NoBinaryMatchingNetwork
from utils.training import fit
from utils.callback import *
from config import PATH
from utils.training import gpu_config
from utils.matching import matching_net_episode
from DataLoader.dataLoaders import get_omniglot_dataloader_v2

# setup_dirs()

##############
# Parameters #
##############
distance = 'l2'
n_train = 1
k_train = 5
q_train = 15
n_test = 1
k_test = 5
q_test = 1

evaluation_episodes = 1000
episodes_per_epoch = 100

n_epochs = 100
dataset_class = OmniglotDataset
num_input_channels = 1

param_str = f'_n={n_train}_k={k_train}_q={q_train}_' \
            f'nv={n_test}_kv={k_test}_qv={q_test}_'\
            f'dist={distance}'

#########
# Model #
#########
model = NoBinaryMatchingNetwork(n_train, k_train, q_train, num_input_channels)
model, use_gpu = gpu_config(model)
model.double()

###########
# Dataset #
###########
# background_taskloader, evaluation_taskloader = get_omniglot_dataloader_v2(episodes_per_epoch, n_train, k_train,
#                                                                          q_train, n_test, k_test, q_test,
#                                                                          dataset_class)
# save dataloader:
# torch.save(background_taskloader, 'background_taskloader.pth')
# torch.save(evaluation_taskloader, 'evaluation_taskloader.pth')
# load dataloader
background_taskloader = torch.load('background_taskloader.pth')
evaluation_taskloader = torch.load('evaluation_taskloader.pth')

"""
############
# Training #
############
print(f'Training Matching Network on Omniglot...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()

callbacks = [
    EvaluateFewShot(
        eval_fn=matching_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=n_test,
        k_way=k_test,
        q_queries=q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(use_gpu, n_test, k_test, q_test),
        distance=distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/matching_nets/{param_str}.pth',
        monitor=f'val_{n_test}-shot_{k_test}-way_acc',
        # monitor=f'val_loss',
    ),
    ReduceLROnPlateau(patience=20, factor=0.5, monitor=f'val_{n_test}-shot_{k_test}-way_acc'),
    CSVLogger(PATH + f'/logs/matching_nets/{param_str}.csv'),
]

fit(
    use_gpu,
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(use_gpu, n_train, k_train, q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=matching_net_episode,
    fit_function_kwargs={'n_shot': n_train, 'k_way': k_train, 'q_queries': q_train, 'train': True, 'distance': distance}
)
"""
import os


PATH = os.path.dirname(os.path.realpath(__file__))

# local DATA_PATH
DATA_PATH = '/home/julien/PycharmProjects/thesis/work/Pytorch/MNIST_Binary_V2/data/'

# colab DATA_PATH
# DATA_PATH = 'data/'


EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
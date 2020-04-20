from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils.core import NShotTaskSampler
from config import PATH


def get_mnist_dataloaders(batch_size_train, batch_size_test):
    """
    create dataloader for MNIST dataset
    :param batch_size_train: int
    :param batch_size_test: int
    :return: DataLoader for train, valid and test data
    """
    # train loader
    train_data = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,),
                                                                                   (0.3081,))]))

    # test_loaer
    test_loader = DataLoader(datasets.MNIST('./data', train=False,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,),
                                                                                               (0.3081,))])),
                             batch_size=batch_size_test, shuffle=False)

    # to split valid data
    n_train_examples = int(len(train_data) * 0.9)
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])
    valid_loader = DataLoader(valid_data, batch_size=batch_size_train, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    print(f'Number of validation examples: {len(valid_data)}')

    print_data_number(train_loader, test_loader)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return train_loader, valid_loader, test_loader, classes


def get_omniglot_dataloaders_classification(batch_size_train, batch_size_test):
    """
    Omniglot data set for one-shot learning but used like classification dataset.
    This dataset contains 1623 different handwritten characters from 50 different alphabets.
    test: 13.180 images
    train: 19.280 images
    image_size = (105, 105)
    :param batch_size_train: int
    :param batch_size_test: int
    :return: train_loader, test_loader
    """
    all_transforms = transforms.Compose([transforms.ToTensor()])

    background_set = datasets.Omniglot(root=PATH + '/data', background=True, download=True, transform=all_transforms)
    # evaluation_set = datasets.Omniglot(root='./data', background=False, download=True, transform=all_transforms)

    # train_loader = DataLoader(background_set, shuffle=True)
    # test_loader = DataLoader(evaluation_set, shuffle=True)

    # to split valid data
    n_train_examples = int(len(background_set) * 0.8)
    n_valid_examples = int((len(background_set) - n_train_examples) * 0.5)
    n_test_examples = len(background_set) - n_train_examples - n_valid_examples
    train_data, valid_data, test_data = random_split(background_set, [n_train_examples, n_valid_examples, n_test_examples])
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_test, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    print_data_number(train_loader, valid_loader)
    return train_loader, valid_loader, test_loader


def get_omniglot_dataloader_v2(episodes_per_epoch, n_train, k_train, q_train, n_test, k_test, q_test, dataset_class):
    """
    Omniglot data set for one-shot learning.
    This function, build dataset for few shot learning task.
    :param episodes_per_epoch:
    :param n_train:
    :param k_train:
    :param q_train:
    :param n_test:
    :param k_test:
    :param q_test:
    :param dataset_class:
    :return:
    """
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, episodes_per_epoch, n_train, k_train, q_train),
        num_workers=4
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, n_test, k_test, q_test),
        num_workers=4
    )
    print_data_number(background_taskloader, evaluation_taskloader)
    return background_taskloader, evaluation_taskloader


def print_data_number(train_loader, test_loader):
    """
    print the number of data in dataloder train and test
    :param train_loader:
    :param test_loader:
    :return: None
    """
    print(f'Number of training examples: {len(train_loader)}')
    print(f'Number of testing examples: {len(test_loader)}')
    return

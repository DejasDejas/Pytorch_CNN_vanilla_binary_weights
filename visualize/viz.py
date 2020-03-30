import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torch import no_grad, max
from torchvision import datasets, transforms


def visualize_activations(model, get_slope):
    epoch = 1
    slope = get_slope(epoch)
    activation = {}
    model.fc1.register_forward_hook(get_activation('fc1', activation))
    model.fc2.register_forward_hook(get_activation('fc2', activation))
    train_ata = get_train_data()
    data, _ = train_ata[0]
    data.unsqueeze_(0)
    model.cpu()
    output = model((data, slope))

    act_conv1 = activation['fc1'].squeeze()
    act_conv2 = activation['fc2'].squeeze()
    fig, axarr = plt.subplots(act_conv1.size(0), figsize=(50, 50))

    print('fc1')
    for idx in range(act_conv1.size(0)):
        axarr[idx].imshow(act_conv1[idx])
    plt.show()

    print('fc2')
    fig, axarr = plt.subplots(act_conv2.size(0), figsize=(50, 50))
    for idx in range(act_conv2.size(0)):
        axarr[idx].imshow(act_conv2[idx])
    plt.show()
    return


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def get_train_data():
    return datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def imshow_v2(inp, title=None):
    if inp.shape[0] == 1:
        inp = inp.reshape((inp.shape[1], inp.shape[2]))
    else:
        inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes):
    out = make_grid(inputs)
    imshow(out)
    plt.show()
    print(classes)


def visualize_model(model, dataloaders, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    with no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def show_som_examples(loader):
    # visualize some example of dataset
    inputs, classes = next(iter(loader))
    show_databatch(inputs, classes)


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

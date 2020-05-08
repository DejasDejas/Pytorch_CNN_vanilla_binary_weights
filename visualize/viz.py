import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid
from torch import no_grad, max
import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from utils.functions import Hardsigmoid
import random
from PIL import Image
# for regions extraction
import collections
from functools import partial
import cv2
from numpy import linalg as LA


class GradientAscent:
    """Provides an interface for activation maximization via gradient descent.
    This class implements the gradient ascent algorithm in order to perform
    activation maximization with convolutional neural networks (CNN).
    `Activation maximization <https://pdfs.semanticscholar.org/65d9/94fb778a8d9e0f632659fb33a082949a50d3.pdf>`_
    is one form of feature visualization that allows us to visualize what CNN
    filters are "looking for", by applying each filter to an input image and
    updating the input image so as to maximize the activation of the filter of
    interest (i.e. treating it as a gradient ascent task with activation as the
    loss). The implementation is inspired by `this demo <https://blog.keras.io/category/demo.html>`_
    by Francois Chollet.
    Args:
        model: A neural network model from `torchvision.models
            <https://pytorch.org/docs/stable/torchvision/models.html>`_,
            typically without the fully-connected part of the network.
            e.g. torchvisions.alexnet(pretrained=True).features
        img_size (int, optional, default=224): The size of an input image to be
            optimized.
        lr (float, optional, default=1.): The step size (or learning rate) of
            the gradient ascent.
        use_gpu (bool, optional, default=False): Use GPU if set to True and
            `torch.cuda.is_available()`.
    """

    ####################
    # Public interface #
    ####################

    def __init__(self, model, nb_channels=3, img_size=224, zoom=False, filter_size=None, lr=1., use_gpu=False):
        self.model = model
        self._img_size = img_size
        self._lr = lr
        self._use_gpu = use_gpu
        self.zoom = zoom
        self.filter_size = filter_size
        self.num_layers = len(list(self.model.named_children()))
        self.activation = None
        self.gradients = None
        self.nb_channels = nb_channels

        self.handlers = []
        self.output = None

        if self.zoom:
            assert self.filter_size != None, 'if zoom, you must choice a filter size'

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu):
        self._use_gpu = use_gpu

    def optimize(self, layer, filter_idx, mean_gradient, ind_x, ind_y, input_=None, num_iter=30):
        """Generates an image that maximally activates the target filter.
        Args:
            layer (torch.nn.modules.conv.Conv2d): The target Conv2d layer from
                which the filter to be chosen, based on `filter_idx`.
            filter_idx (int): The index of the target filter.
            num_iter (int, optional, default=30): The number of iteration for
                the gradient ascent operation.
        Returns:
            output (list of torch.Tensor): With dimentions
                :math:`(num_iter, C, H, W)`. The size of the image is
                determined by `img_size` attribute which defaults to 224.
        """

        # Validate the type of the layer

        if type(layer) != nn.modules.conv.Conv2d:
            raise TypeError('The layer must be nn.modules.conv.Conv2d.')

        # Validate filter index

        num_total_filters = layer.out_channels
        self._validate_filter_idx(num_total_filters, filter_idx)

        # Inisialize input (as noise) if not provided

        if input_ is None:
            if self.nb_channels==3:
                input_ = np.uint8(np.random.uniform(
                0, 255, (self._img_size, self._img_size, 3))/255)
            else:
                input_ = np.uint8(np.random.uniform(
                    0, 255, (self._img_size, self._img_size, 1))/255)
            input_ = apply_transforms(input_, size=self._img_size)

        if torch.cuda.is_available() and self.use_gpu:
            self.model = self.model.to('cuda')
            input_ = input_.to('cuda')

        # Remove previous hooks if any
        while len(self.handlers) > 0:
            self.handlers.pop().remove()

        # Register hooks to record activation and gradients

        self.handlers.append(self._register_forward_hooks(layer, filter_idx, mean_gradient, ind_x, ind_y))
        self.handlers.append(self._register_backward_hooks())

        # Inisialize gradients

        self.gradients = torch.zeros(input_.shape)

        # Optimize

        return self._ascent(input_, num_iter)

    def visualize(self, layer, MNIST, filter_idxs=None, mean_gradient=True, ind_x=None, ind_y=None,
                  first_conv_layer=False, lr=1., num_iter=30,
                  num_subplots=4, figsize=(4, 4), title='Conv2d',
                  return_output=False):
        """Optimizes for the target layer/filter and visualizes the output.
        A method that combines optimization and visualization. There are
        mainly 3 types of operations, given a target layer:
        1. If `filter_idxs` is provided as an integer, it optimizes for the
            filter specified and plots the output.
        2. If `filter_idxs` is provided as a list of integers, it optimizes for
            all the filters specified and plots the output.
        3. if `filter_idx` is not provided, i.e. None, it randomly chooses
            `num_subplots` number of filters from the layer provided and
            plots the output.
        It also returns the output of the optimization, if specified with
        `return_output=True`.
        Args:
            layer (torch.nn.modules.conv.Conv2d): The target Conv2d layer from
                which the filter to be chosen, based on `filter_idx`.
            filter_idxs (int or list of int, optional, default=None): The index
                or indecies of the target filter(s).
            lr (float, optional, default=.1): The step size of optimization.
            num_iter (int, optional, default=30): The number of iteration for
                the gradient ascent operation.
            num_subplots (int, optional, default=4): The number of filters to
                optimize for and visualize. Relevant in case 3 above.
            figsize (tuple, optional, default=(4, 4)): The size of the plot.
                Relevant in case 1 above.
            title (str, optional default='Conv2d'): The title of the plot.
            return_output (bool, optional, default=False): Returns the
                output(s) of optimization if set to True.
        Returns:
            For a single optimization (i.e. case 1 above):
                output (list of torch.Tensor): With dimentions
                    :math:`(num_iter, C, H, W)`. The size of the image is
                    determined by `img_size` attribute which defaults to 224.
            For multiple optimization (i.e. case 2 or 3 above):
                output (list of list of torch.Tensor): With dimentions
                    :math:`(num_subplots, num_iter, C, H, W)`. The size of the
                    image is determined by `img_size` attribute which defaults
                    to 224.
        """
        
        self._lr = lr
        self.mean_gradient = mean_gradient
        self.ind_x = ind_x
        self.ind_y = ind_y
        self.first_conv_layer = first_conv_layer
        

        if not self.mean_gradient:
            assert self.ind_x is not None and self.ind_y is not None, 'if mean_gradient is false, you must choice x ' \
                                                                      'and y index '

        if type(filter_idxs) == int:
            self._visualize_filter(layer, MNIST,
                                   filter_idxs,
                                   self.mean_gradient,
                                   self.ind_x,
                                   self.ind_y,
                                   self.first_conv_layer,
                                   num_iter=num_iter,
                                   figsize=figsize,
                                   title=title)
        else:
            num_total_filters = layer.out_channels

            if filter_idxs is None:
                num_subplots = min(num_total_filters, num_subplots)
                filter_idxs = np.random.choice(range(num_total_filters),
                                               size=num_subplots)

            self._visualize_filters(layer, MNIST,
                                    filter_idxs,
                                    self.mean_gradient,
                                    self.ind_x,
                                    self.ind_y,
                                    self.first_conv_layer,
                                    num_iter,
                                    len(filter_idxs),
                                    title=title)
        if return_output:
            return self.output

    #####################
    # Private interface #
    #####################

    def _register_forward_hooks(self, layer, filter_idx, mean_gradient, ind_x, ind_y):
        def _record_activation(module, input_, output):
            if mean_gradient:
                # maximization of mean for filter_idx
                self.activation = torch.mean(output[:, filter_idx, :, :])
            else:
                # maximization of a specific neuron for filter_idx
                self.activation = output[:, filter_idx, ind_x, ind_y]

        return layer.register_forward_hook(_record_activation)

    def _register_backward_hooks(self):
        def _record_gradients(module, grad_in, grad_out):
            if self.gradients.shape == grad_in[0].shape:
                self.gradients = grad_in[0]

        for _, module in self.model.named_modules():
            if self.nb_channels==3:
                if isinstance(module, nn.modules.conv.Conv2d) and \
                        module.in_channels == 3:
                    return module.register_backward_hook(_record_gradients)
            else:
                if isinstance(module, nn.modules.conv.Conv2d) and \
                        module.in_channels == 1:
                    return module.register_backward_hook(_record_gradients)                

    def _ascent(self, x, num_iter):
        output = []
        for i in range(num_iter):
            self.model(x)
            self.activation.backward()
            self.gradients /= (torch.sqrt(torch.mean(
                torch.mul(self.gradients, self.gradients))) + 1e-5)
            x = x + self.gradients * self._lr
            output.append(x)
        return output

    def _validate_filter_idx(self, num_filters, filter_idx):
        if not np.issubdtype(type(filter_idx), np.integer):
            raise TypeError('Indecies must be integers.')
        elif (filter_idx < 0) or (filter_idx > num_filters):
            raise ValueError(f'Filter index must be between 0 and {num_filters - 1}.')

    def _visualize_filter(self, layer, MNIST, filter_idx, mean_gradient, ind_x, ind_y, first_conv_layer, num_iter, figsize, title):
        self.output = self.optimize(layer, filter_idx, mean_gradient, ind_x, ind_y, num_iter=num_iter)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.title(title)

        plt.imshow(format_for_plotting(
            standardize_and_clip(self.output[-1],
                                 MNIST,
                                 saturation=0.15,
                                 brightness=0.7)), cmap='gray');

        if self.zoom and first_conv_layer:
            xmin = ind_x * 2
            xmax = (ind_x * 2) + self.filter_size
            ymin = ind_y * 2
            ymax = (ind_y * 2) + self.filter_size
            plt.axis([xmin, xmax, ymin, ymax])

        plt.show()
        # plt.imsave('plot_image_maximize_filter_layer2_model_MNIST.png')

    def _visualize_filters(self, layer, MNIST, filter_idxs, mean_gradient, ind_x, ind_y, first_conv_layer, num_iter, num_subplots, title):
        # Prepare the main plot

        num_cols = 4
        num_rows = int(np.ceil(num_subplots / num_cols))

        fig = plt.figure(figsize=(16, num_rows * 5))
        plt.title(title)
        plt.axis('off')

        self.output = []

        # Plot subplots
        for i, filter_idx in enumerate(filter_idxs):
            output = self.optimize(layer, filter_idx, mean_gradient, ind_x, ind_y, num_iter=num_iter)

            self.output.append(output)

            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'filter {filter_idx}')

            ax.imshow(format_for_plotting(
                standardize_and_clip(output[-1],
                                     MNIST,
                                     saturation=0.15,
                                     brightness=0.7)), cmap='gray')
            if self.zoom and first_conv_layer:
                xmin = (ind_x * 2)
                xmax = (ind_x * 2)
                ymin = (ind_y * 2)
                ymax = (ind_y * 2)
                ax.axis([xmin, xmax, ymin, ymax])

        plt.subplots_adjust(wspace=0, hspace=0);
        # plt.imsave('plot_image_maximize_filter_layer2_model_MNIST.png')


def viz_activations(model, loader, index_data=None):
    activation = {}

    for name, m in model.named_modules():
        if type(m) == Hardsigmoid or type(m) == nn.ReLU:
            m.register_forward_hook(get_activation(name, activation))

    if index_data == None:
        index_data = random.randint(0, len(loader))

    dataiter = iter(loader)
    images, labels = dataiter.next()
    image = images[index_data].unsqueeze(0)
    label = labels[index_data].item()

    model.cpu()
    output = model(image)

    for keys in activation:
        act_conv = activation[keys].squeeze()
        print('{} for label {}'.format(keys, label))
        visTensor(act_conv.reshape(act_conv.shape[0], 1, act_conv.shape[1], act_conv.shape[2]), ch=0, allkernels=False)
        plt.show()
        """
      fig, axarr = plt.subplots(act_conv.size(0), figsize=(50,50))
      for idx in range(act_conv.size(0)):
        axarr[idx].imshow(act_conv[idx])
      plt.show()
      """


def viz_filters(model):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            filters = m.weight.data.clone()
            visTensor(filters.cpu(), ch=0, allkernels=False)
            plt.ioff()
            print('Visualization filters learned for layer: {}'.format(name))
            plt.show()


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def viz_heatmap(model, name_model, loader, index_data=None, save=True):
    activation = {}
    for name, m in model.named_modules():
        if type(m) == Hardsigmoid or type(m) == nn.ReLU:
            m.register_forward_hook(get_activation(name, activation))

    if index_data == None:
        index_data = random.randint(0, len(loader))

    dataiter = iter(loader)
    images, labels = dataiter.next()
    image = images[index_data].unsqueeze(0)
    label = labels[index_data].item()

    model.cpu()
    output = model(image)

    for keys in activation:
        heatmap = torch.mean(activation[keys], dim=0)[0].squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        print('layer:{} :heatrmap for an image of label {} with model {}'.format(keys, label, name_model))
        fig, ax = plt.subplots()
        cs = ax.matshow(heatmap.squeeze())
        cbar = fig.colorbar(cs)
        plt.show()
        if save:
            plt.imsave('results/MNIST_results/heatmap_png/heatmap' +
                       name_model + name + '.png', heatmap)


def test_predict_few_examples(model, loader):
    # classes of fashion mnist dataset
    # classes = ['0','1','2','3','4','5','6','7','8','9']
    # creating iterator for iterating the dataset
    dataiter = iter(loader)
    images, labels = dataiter.next()
    images_arr = []
    labels_arr = []
    pred_arr = []
    # moving model to cpu for inference 
    model.to("cpu")
    # iterating on the dataset to predict the output
    for i in range(0, 10):
        images_arr.append(images[i].unsqueeze(0))
        labels_arr.append(labels[i].item())
        ps = torch.exp(model(images_arr[i]))
        ps = ps.data.numpy().squeeze()
        pred_arr.append(np.argmax(ps))
    # plotting the results
    fig = plt.figure(figsize=(25, 4))
    for i in range(10):
        ax = fig.add_subplot(2, 20 / 2, i + 1, xticks=[], yticks=[])
        ax.imshow(images_arr[i].resize_(1, images[0].shape[-1], images[0].shape[-2]).numpy().squeeze(), cmap='gray')
        ax.set_title("{} ({})".format(pred_arr[i], labels_arr[i]),
                     color=("green" if pred_arr[i] == labels_arr[i] else "red"))


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


def imshow_v3(img, title):
    """Custom function to display the image using matplotlib"""

    # define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # convert the tensor img to numpy img and de normalize
    npimg = np.multiply(img, std_correction) + mean_correction

    # plot the numpy image
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


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


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = make_grid(tensor, nrow=nrow, normalize=True, padding=padding, pad_value=1)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def apply_transforms(image, size=224):
    """Transforms a PIL image to torch.Tensor.
    Applies a series of tranformations on PIL image including a conversion
    to a tensor. The returned tensor has a shape of :math:`(N, C, H, W)` and
    is ready to be used as an input to neural networks.
    First the image is resized to 256, then cropped to 224. The `means` and
    `stds` for normalisation are taken from numbers used in ImageNet, as
    currently developing the package for visualizing pre-trained models.
    The plan is to to expand this to handle custom size/mean/std.
    Args:
        image (PIL.Image.Image or numpy array)
        size (int, optional, default=224): Desired size (width/height) of the
            output tensor
    Shape:
        Input: :math:`(C, H, W)` for numpy array
        Output: :math:`(N, C, H, W)`
    Returns:
        torch.Tensor (torch.float32): Transformed image tensor
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    if not isinstance(image, Image.Image):
        image = F.to_pil_image(image)

    # means = [0.485, 0.456, 0.406]
    # stds = [0.229, 0.224, 0.225]
    # to only one channel
    means = [0.1307]
    stds = [0.3081]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds)
    ])

    tensor = transform(image).unsqueeze(0)

    tensor.requires_grad = True

    return tensor


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """
    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def standardize_and_clip(tensor, MNIST, min_value=0.0, max_value=1.0,
                         saturation=0.1, brightness=0.5):
    """Standardizes and clips input tensor.
    Standardizes the input tensor (mean = 0.0, std = 1.0). The color saturation
    and brightness are adjusted, before tensor values are clipped to min/max
    (default: 0.0/1.0).
    Args:
        tensor (torch.Tensor):
        min_value (float, optional, default=0.0)
        max_value (float, optional, default=1.0)
        saturation (float, optional, default=0.1)
        brightness (float, optional, default=0.5)
    Shape:
        Input: :math:`(C, H, W)`
        Output: Same as the input
    Return:
        torch.Tensor (torch.float32): Normalised tensor with values between
            [min_value, max_value]
    """

    tensor = tensor.detach().cpu()

    # mean = tensor.mean()
    # std = tensor.std()
    mean = 0.1307
    std = 0.3081
    
    if std == 0:
        std += 1e-7

    # standardized = tensor.sub(mean).div(std).mul(saturation)
    if MNIST:
        unstandardized = tensor.mul(std).add(mean).mul(saturation)
    else:
        unstandardized = tensor.mul(saturation)
    # standardized = tensor.mul(saturation)
    clipped = unstandardized.add(brightness).clamp(min_value, max_value)

    return clipped


#####################################################
# modules for extract regions that maximizes filters#
#####################################################


def get_region_layer1(image, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w, return_all=False):

    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        reduice_region_col_size = index_col_hl
        index_col_hl = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl < 0:
        reduice_region_raw_size = index_raw_hl
        index_raw_hl = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl
    end_col = index_col_hl + filter_size + reduice_region_col_size
    begin_raw = index_raw_hl
    end_raw = index_raw_hl + filter_size + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col,begin_raw:end_raw]
    if region.shape != (filter_size, filter_size):
        region = cv2.resize(region, (filter_size, filter_size), interpolation=cv2.INTER_AREA)

    if return_all:
        return region, begin_col, end_col, begin_raw, end_raw
    else:
        return region



def get_region_layer2(image, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w):

    region_shape = 7
    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        reduice_region_col_size = index_col_hl_2
        index_col_hl_2 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_2 < 0:
        reduice_region_raw_size = index_raw_hl_2
        index_raw_hl_2 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_2
    end_col = index_col_hl_2 + ((filter_size * stride) + 1) + reduice_region_col_size
    begin_raw = index_raw_hl_2
    end_raw = index_raw_hl_2 + ((filter_size * stride) + 1) + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col,begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    return region


def get_filter_layer2():
    return np.array(([[1, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 1]]))


def get_region_layer3(image, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w):

    region_shape = 15
    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        index_col_hl_2 = 0
    if index_raw_hl_2 < 0:
        index_raw_hl_2 = 0

    index_col_hl_3 = (index_col_hl_2 * stride) - padding
    index_raw_hl_3 = (index_raw_hl_2 * stride) - padding

    if index_col_hl_3 < 0:
        reduice_region_col_size = index_col_hl_3
        index_col_hl_3 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_3 < 0:
        reduice_region_raw_size = index_raw_hl_3
        index_raw_hl_3 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_3
    end_col = index_col_hl_3 + ((((filter_size * stride) + 1) * stride) + 1) + reduice_region_col_size
    begin_raw = index_raw_hl_3
    end_raw = index_raw_hl_3 + ((((filter_size * stride) + 1) * stride) + 1) + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col,begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    return region


def get_filter_layer3():
    return np.array(([[1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]]))


def get_region_layer4(image, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w):

    region_shape = 31
    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        index_col_hl_2 = 0
    if index_raw_hl_2 < 0:
        index_raw_hl_2 = 0

    index_col_hl_3 = (index_col_hl_2 * stride) - padding
    index_raw_hl_3 = (index_raw_hl_2 * stride) - padding

    if index_col_hl_3 < 0:
        index_col_hl_3 = 0
    if index_raw_hl_3 < 0:
        index_raw_hl_3 = 0

    index_col_hl_4 = (index_col_hl_3 * stride) - padding
    index_raw_hl_4 = (index_raw_hl_3 * stride) - padding

    if index_col_hl_4 < 0:
        reduice_region_col_size = index_col_hl_4
        index_col_hl_4 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_4 < 0:
        reduice_region_raw_size = index_raw_hl_4
        index_raw_hl_4 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_4
    end_col = index_col_hl_4 + (((((filter_size * stride) + 1) * stride) + 1) * stride) + 1 + reduice_region_col_size
    begin_raw = index_raw_hl_4
    end_raw = index_raw_hl_4 + (((((filter_size * stride) + 1) * stride) + 1) * stride) + 1 + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col,begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    return region


def get_filter_layer4():
    return np.array(([[1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [2, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 2],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1],
                      [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1]]))


def get_all_regions_max(loader, activations, stride, padding, filter_size, len_img_h, len_img_w):

    dataiter = iter(loader)
    images, _ = dataiter.next()
    print('nb images: {}'.format(len(images)))

    print('begin extraction regions')
    region_final = {}
    activation_final = {}
    activation_final_normalized = {}

    filter_layer2 = get_filter_layer2()
    filter_layer3 = get_filter_layer3()
    filter_layer4 = get_filter_layer4()

    for name, fm in activations.items():
        # for each image of fm
        if name == 'layer1':
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], 3, 3))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        if name == 'layer2':
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], 7, 7))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        if name == 'layer3':
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], 15, 15))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
        if name == 'layer4':
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], 31, 31))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
        for j in range(fm.shape[0]):
            print('treating image n {}/{}, for layer: {}'.format(j, fm.shape[0], name))

            im = images[j].unsqueeze(0).numpy().squeeze()  # image i of batch batch: numpy array: (28,28)
            if name == 'layer1':
                regions_im_j = np.zeros((fm.shape[1], 3, 3))  # initialise empty list of regions for batch batch
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            if name == 'layer2':
                regions_im_j = np.zeros((fm.shape[1], 7, 7))
                regions_im_j = (regions_im_j * filter_layer2) / 4
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            if name == 'layer3':
                regions_im_j = np.zeros((fm.shape[1], 15, 15))
                regions_im_j = (regions_im_j * filter_layer3) / 4
                activation_im_j = np.zeros((fm.shape[1]))
            if name == 'layer4':
                regions_im_j = np.zeros((fm.shape[1], 31, 31))
                regions_im_j = (regions_im_j * filter_layer4) / 4
                activation_im_j = np.zeros((fm.shape[1]))
            for i in range(fm.shape[1]):  # for all fm in image j
                # act_max = max(fm[j][i], key=abs)  # get max abs activation value in fm j
                act_max = fm[j][i].max()
                ind_x = int((np.where(fm[j][i] == act_max)[0])[0])  # get index (x,y) of act_max
                ind_y = int((np.where(fm[j][i] == act_max)[1])[0])

                if name == 'layer1':
                    region = get_region_layer1(im, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w)
                if name == 'layer2':
                    region = get_region_layer2(im, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w)
                if name == 'layer3':
                    region = get_region_layer3(im, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w)
                if name == 'layer4':
                    region = get_region_layer4(im, ind_x, ind_y, name, stride, padding, filter_size, len_img_h, len_img_w)

                regions_im_j[i] = region
                activation_im_j[i] = act_max.detach().numpy()
                norme = LA.norm(region, 1)
                if norme == 0.0:
                    print('norm null')
                    activation_im_j_normalized[i]=act_max.detach().numpy()
                else:
                    activation_im_j_normalized[i] = act_max.detach().numpy()/norme
                # print('region values: {}, with activation: {} and norme for this region: {} and result: act/norm: {}'.format(region, act_max, LA.norm(region, 1), activation_im_j_normalized[i]))
            regions_layer[j] = regions_im_j
            activation_layer[j] = activation_im_j
            activation_layer_normalized[j] = activation_im_j_normalized
        region_final[name] = regions_layer
        activation_final[name] = activation_layer
        activation_final_normalized[name]= activation_layer_normalized

    return region_final, activation_final, activation_final_normalized


#####################################
# Modules for viz regions of interest#
#####################################


def get_regions_interest(regions, activation, activations_normalized, best, worst, viz_mean_img, viz_grid,   percentage=None, list_filter=None, nrow=8):
    """
  get regions of interest
  """
    nb_filter = activation.shape[1]

    if best == False and worst == False:
        assert percentage != None, "if don't choice best or worst value, you didn't choice a percentage value"
    if best == True and worst == True:
        raise TypeError('choice only one value at True between best an worst')

    # consider only regions of all image of list_filter or all filter
    if list_filter == None:
        print('Interest of all filters')
        regions_interest_filter = regions
        activations_values_interest = activation
        activations_values_interest_normalized = activations_normalized
        nb_filter = nb_filter
    else:
        # assert max(list_filter) < nb_filter and min(list_filter) >= 0, 'filter choisen out of range'
        print('Interest of filters:', list_filter)
        regions_interest_filter = get_index_filter_interest(regions, list_filter)
        activations_values_interest = activation[:, list_filter]
        activations_values_interest_normalized = activations_normalized[:, list_filter]
        nb_filter = len(list_filter)

    # consider a percent of best or worst activations:
    if percentage == None:
        print('Consider all image regions')
        selected_regions = regions_interest_filter
        activation_values = activation[:, list_filter]
        activation_values_normalized = activations_normalized[:, list_filter]
    else:
        assert percentage <= 100 and percentage >= 0, 'percentage value must be in 0 and 100'
        n = int((len(activation) * percentage) / 100)
        print('Consider {}% image regions = {} images'.format(percentage, n))
        selected_regions, activation_values = get_n_first_regions_index(best, worst, n, activations_values_interest, nb_filter,
                                                     regions_interest_filter)
        selected_regions_normalized, activation_values_normalized = get_n_first_regions_index(best, worst, n, activations_values_interest_normalized, nb_filter,
                                                     regions_interest_filter)

    nb_regions = selected_regions[0].shape[0]

    # visualization: one mean image or grid image:
    if viz_mean_img:
        nb_image = 1
        print('mean image:')
        for i, ind_filter in enumerate(list_filter):
            print('mean regions of {} regions more={} or worst={} active for filter number: {} :'.format(n, best, worst,
                                                                                                         ind_filter))
            mean_img = np.mean(selected_regions[i], 0)
            # mean_img_1 = (mean_img - np.min(mean_img))/(np.max(mean_img)-np.min(mean_img))
            # mean_img_2 = (mean_img*0.3081)+0.1307
            
            # viz_regions(nb_image, mean_img_2[range(mean_img_2.shape[0]-1,-1,-1),:], nrow)  #inverse image
            # viz_regions(nb_image, mean_img_2, nrow)
            # viz_regions(nb_image, mean_img_1, nrow)
            viz_regions(nb_image, mean_img, nrow)
            plt.show()
            print('normalized region:')
            mean_img_normalized = np.mean(selected_regions_normalized[i], 0)
            viz_regions(nb_image, mean_img_normalized, nrow)
            plt.show()
            


    if viz_grid:
        nb_image = nb_regions
        print('grid image')
        for i, ind_filter in enumerate(list_filter):
            region_to_print = []
            print('grid regions of {} regions more={} or worst={} active for filter number: {} :'.format(n, best, worst,
                                                                                                         ind_filter))
            for j in range(nb_regions):
                region_to_print.append(selected_regions[i][j])
            viz_regions(nb_image, region_to_print, nrow)
            plt.show()
            
            print('normalized regions:')
            region_to_print_normalized = []
            for j in range(nb_regions):
                region_to_print_normalized.append(selected_regions_normalized[i][j])
            viz_regions(nb_image, region_to_print_normalized, nrow)
            plt.show()

    return selected_regions, activation_values, activation_values_normalized


def viz_regions(nb_image, regions, nrow):
    """
  visualize region of interest
  """
    regions = torch.tensor(regions)
    regions = regions.reshape((nb_image, 1, regions.shape[-2], regions.shape[-1]))
    visTensor(regions, ch=0, allkernels=False, nrow=nrow)
    plt.ioff()
    plt.show()


def get_n_first_regions_index(best, worst, n, activation, nb_filter, regions):
    """
  select only regions that we want 
  """
    regions_selected = []
    activation_values = []
    if best:
        for i in range(nb_filter):
            ind_filter = (-activation[:, i]).argsort()[:n]
            regions_selected.append(regions[ind_filter, i])
            activation_values.append(activation[ind_filter, i])
        return regions_selected, activation_values

    elif worst:
        for i in range(nb_filter):
            ind_filter = activation[:, i].argsort()[:n]
            regions_selected.append(regions[ind_filter, i])
            activation_values.append(activation[ind_filter, i])
        return regions_selected, activation_values

    else:
        print('choice worst or best with bool True or False')


def get_index_filter_interest(regions, list_filter):
    """
  extract only regions of the filter interest
  """
    return regions[:, list_filter]

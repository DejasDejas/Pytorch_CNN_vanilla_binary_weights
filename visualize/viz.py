import matplotlib.pyplot as plt
import numpy as np
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


def viz_activations(model, loader, index_data=None):
  activation = {}

  for name, m in model.named_modules():
    if type(m)==Hardsigmoid:
      m.register_forward_hook(get_activation(name, activation))

  if index_data==None:
    index_data = random.randint(0,len(loader))

  dataiter = iter(loader)
  images, labels = dataiter.next()
  image = images[index_data].unsqueeze(0)
  label = labels[index_data].item()

  model.cpu()
  output = model(image)

  for keys in activation:
      act_conv = activation[keys].squeeze()
      print('{} for label {}'.format(keys, label))
      visTensor(act_conv.reshape(act_conv.shape[0],1,act_conv.shape[1],act_conv.shape[2]), ch=0, allkernels=False)
      plt.show()
      """
      fig, axarr = plt.subplots(act_conv.size(0), figsize=(50,50))
      for idx in range(act_conv.size(0)):
        axarr[idx].imshow(act_conv[idx])
      plt.show()
      """

def viz_filters(model):

  for name, m in model.named_modules():
    if type(m)==nn.Conv2d:
      filters = m.weight.data.clone()
      visTensor(filters.cpu(), ch=0, allkernels=False)
      plt.ioff()
      print('Visualization filters learned for layer: {}'.format(name))
      plt.show()
      
      
def viz_heatmap(model, name_model, loader, index_data=None, save=True):

    activation = {}
    for name, m in model.named_modules():
      if type(m)==Hardsigmoid:
        m.register_forward_hook(get_activation(name, activation))

    if index_data==None:
      index_data = random.randint(0,len(loader))

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
    classes = ['0','1','2','3','4','5','6','7','8','9']
    # creating iterator for iterating the dataset
    dataiter = iter(loader)
    images, labels = dataiter.next()
    images_arr = []
    labels_arr = []
    pred_arr = []
    # moving model to cpu for inference 
    model.to("cpu")
    # iterating on the dataset to predict the output
    for i in range(0,10):
        images_arr.append(images[i].unsqueeze(0))
        labels_arr.append(labels[i].item())
        ps = torch.exp(model(images_arr[i]))
        ps = ps.data.numpy().squeeze()
        pred_arr.append(np.argmax(ps))
    # plotting the results
    fig = plt.figure(figsize=(25,4))
    for i in range(10):
        ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
        ax.imshow(images_arr[i].resize_(1, images[0].shape[-1],  images[0].shape[-2]).numpy().squeeze(), cmap='gray')
        ax.set_title("{} ({})".format(classes[pred_arr[i]], classes[labels_arr[i]]),
                    color=("green" if pred_arr[i]==labels_arr[i] else "red"))


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
  
  #define std correction to be made
  std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
  
  #define mean correction to be made
  mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
  
  #convert the tensor img to numpy img and de normalize 
  npimg = np.multiply(img, std_correction) + mean_correction
  
  #plot the numpy image
  plt.figure(figsize = (10, 10))
  plt.axis("off")
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.title(title)
  plt.show()


def show_databatch(inputs, classes):
    out = make_grid(inputs)
    imshow(out)
    plt.show()
    print(classes)
    

def show_batch_images(dataloader):

  images,_ = next(iter(dataloader))
  
  #run the model on the images
  outputs = model((images, 1.0))
  
  #get the maximum class 
  _, pred = torch.max(outputs.data, 1)
  
  #make grid
  img = torchvision.utils.make_grid(images)
  
  #call the function
  imshow(img, title=[classes[x.item()] for x in pred])
  
  return images, pred
  
  
#custom function to fetch images from dataloader
def show_simple_image(dataloader):
  images,_ = next(iter(dataloader))
  
  #run the model on the images
  outputs = model((images, 1.0))
  
  #get the maximum class 
  _, pred = torch.max(outputs.data, 1)

  
  #call the function
  # imshow(images[0], title=[classes[x.item()] for x in pred])
  
  return images[0]
  

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
    means = [0.406]
    stds = [0.225]

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
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
        
        
        
def standardize_and_clip(tensor, min_value=0.0, max_value=1.0,
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

    mean = tensor.mean()
    std = tensor.std()

    if std == 0:
        std += 1e-7

    standardized = tensor.sub(mean).div(std).mul(saturation)
    clipped = standardized.add(brightness).clamp(min_value, max_value)

    return clipped


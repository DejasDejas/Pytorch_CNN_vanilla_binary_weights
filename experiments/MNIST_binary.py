import sys
sys.path.append('')

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from utils.models import get_my_model_MNIST, fetch_last_checkpoint_model_filename
from DataLoader.dataLoaders import get_mnist_dataloaders
from utils.functions import Hardsigmoid

from visualize.viz import visTensor, get_activation, viz_activations, viz_filters
from visualize.viz import viz_heatmap, test_predict_few_examples, standardize_and_clip, format_for_plotting
from visualize.viz import apply_transforms, GradientAscent, get_filter_layer2, get_labels_histogram
from visualize.viz import get_region_layer1, get_region_layer2, get_regions_interest, get_all_regions_max
from visualize.viz import compare_two_histograms,viz_regions
from torchvision import datasets, transforms

# for regions extraction
import collections
from functools import partial
import cv2
import skimage.measure
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import time
import pickle

batch_size_train = 10000
batch_size_test = 10000
# Dataset
train_loader, valid_loader, test_loader, classes = get_mnist_dataloaders(batch_size_train, batch_size_test)

# load model pre trained no binary
binary = False
maxpooling = False

model_no_binary_stride, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling)

path_model = 'trained_models/MNIST/No_binary_models/with_bias/stride'
if torch.cuda.is_available():
  model_no_binary_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_no_binary_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))
print("Model Loaded", name_model)

# load model pre trained no binary
binary = False
maxpooling = True

model_no_binary_maxpooling, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling)

path_model = 'trained_models/MNIST/No_binary_models/with_bias/maxpooling'
if torch.cuda.is_available():
  model_no_binary_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_no_binary_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))
print("Model Loaded", name_model)

# load model pre trained binary
binary = True    
maxpooling = False
model_binary_stride, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling)

path_model = 'trained_models/MNIST/Binary_models/with_bias/stride'
if torch.cuda.is_available():
  model_binary_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_binary_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))
print("Model Loaded", name_model)

# load model pre trained binary
binary = True    
maxpooling = True
model_binary_maxpooling, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling)

path_model = 'trained_models/MNIST/Binary_models/with_bias/maxpooling'
if torch.cuda.is_available():
  model_binary_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_binary_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))
print("Model Loaded", name_model)

# load model pre trained binary
binary = False
mixt = True    
maxpooling = False

model_mixt_stride, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling, mixt=mixt)

path_model = 'trained_models/MNIST/Mixt_models/stride'
if torch.cuda.is_available():
  model_mixt_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_mixt_stride.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))

print("Model Loaded", name_model)

# load model pre trained binary
binary = False
mixt = True    
maxpooling = True

model_mixt_maxpooling, name_model = get_my_model_MNIST(binary, maxpooling=maxpooling, mixt=mixt)

path_model = 'trained_models/MNIST/Mixt_models/maxpooling'
if torch.cuda.is_available():
  model_mixt_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model)))
else:
  model_mixt_maxpooling.load_state_dict(torch.load(fetch_last_checkpoint_model_filename(path_model), map_location=torch.device('cpu')))

print("Model Loaded", name_model)

MNIST_dataset_classe_all = np.load('results/MNIST_results/dataset_sorted/MNIST_dataset_classe_all.npy', allow_pickle=True)


def get_features(model, index, data):

  features = []
  feature_extractor = torch.nn.Sequential(*list(model.children())[:index+1])

  for i in range(len(data)):
    features.append(feature_extractor(data[i]))
  return features

def get_features_mixt_model(model, index, data):

  features_no_binary = []
  features_binary = []
  feature_extractor_no_binary = torch.nn.Sequential(*list(model.children())[:5])
  feature_extractor_binary = torch.nn.Sequential(*list(model.children())[6:11])

  for i in range(len(data)):
    features_no_binary.append(feature_extractor_no_binary(data[i]))
    features_binary.append(feature_extractor_binary(data[i]))
  return features_no_binary, features_binary


def get_average_representation(max_pooling, extracted_features, data_test):
    
    mp_size = max_pooling
    feature_mean_flattened = []
    features = extracted_features

    for i in range(len(features)):
      features[i] = skimage.measure.block_reduce(features[i].detach().numpy(), (1, 1, mp_size, mp_size), np.max)
      
    for i in range(len(features)):
      if data_test:
        feature_mean_flattened.append(features[i].reshape((1, features[0].shape[0], features[0].shape[1]*features[0].shape[2]*features[0].shape[3])))
      else:
        mean = np.mean(features[i], 0)
        feature_mean_flattened.append(np.ndarray.flatten(mean))
    if data_test:
      feature_mean_flattened = feature_mean_flattened[0][0]

    feature_mean_flattened = torch.FloatTensor(feature_mean_flattened)
    return feature_mean_flattened


def clf(X_train, y_train, X_test, y_test):
  
  start = time.time()
  clf = NearestCentroid()
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  clf_score = np.round(score*100, 3)
  stop = time.time()

  execute_timing_clf = np.round(stop - start, 3)
  return clf_score, execute_timing_clf


def knn(X_train, y_train, X_test, y_test, n_neighbors):

  start = time.time()
  knn = KNeighborsClassifier(n_neighbors=n_neighbors)
  knn.fit(X_train, y_train)
  score = knn.score(X_test, y_test)
  knn_score = np.round(score*100, 3)
  stop = time.time()

  execute_timing_knn = np.round(stop - start, 3)
  return knn_score, execute_timing_knn


def compute_features_all_classes(model, index_layer, data_all_classes, mixt_model=False):
  
  if mixt_model:
    features_all_classes_no_binary, features_all_classes_binary = get_features_mixt_model(model, index_layer, data_all_classes)
    fm_shape = features_all_classes_no_binary[0].shape[-1]
    return features_all_classes_no_binary, features_all_classes_binary, fm_shape
  else:
    features_all_classes = get_features(model, index_layer, data_all_classes)
    fm_shape = features_all_classes[0].shape[-1]
    return features_all_classes, fm_shape


def compute_average_representation_all_classes(features_all_classes, maxpooling_size):
  
  average_representation_all_classes = get_average_representation(maxpooling_size, features_all_classes, data_test=False)
  return average_representation_all_classes


def compute_score_onn(model, average_representation_all_classes, data, index, maxpooling_size, n_neighbors, nearest_centroid_score, knn_score, return_score_clf=False, mixt_model=False):

  if mixt_model:
    features_data_no_binary, features_data_binary = get_features_mixt_model(model, index, data)
    average_representation_all_classes_no_binary = get_average_representation(maxpooling_size, features_data_no_binary, data_test=True)
    average_representation_all_classes_binary = get_average_representation(maxpooling_size, features_data_binary, data_test=True)
    average_representation_all_classes_concatenate = torch.cat((average_representation_all_classes_no_binary, average_representation_all_classes_binary), 1)
    X_test = [average_representation_all_classes_no_binary, average_representation_all_classes_binary, average_representation_all_classes_concatenate]
  else:
    # compute data representation with maxpooling_size:
    features_data = get_features(model, index, data)
    average_representation_data = get_average_representation(maxpooling_size, features_data, data_test=True)
    X_test = average_representation_data

  # compute score with clf and knn:
  # parameters:
  X_train = average_representation_all_classes
  y_train = range(10)
  y_test = labels
  dim = len(X_train[0])
  n_classes = len(np.unique(y_train))

  if mixt_model:
    clf_score = []
    knn_score = []
    for i, x_test in enumerate(X_test):
      # compute clf score:
      if nearest_centroid_score:
        clf_score.append(clf(X_train[i], y_train, x_test, y_test)[0])
      # compute knn score:
      if knn_score:
        knn_score.append(knn(X_train[i], y_train, x_test, y_test, n_neighbors)[0])
  else: 
    # compute clf score:
    if nearest_centroid_score:
      clf_score, execute_timing_clf = clf(X_train, y_train, X_test, y_test)
    # compute knn score:
    if knn_score:
      knn_score, execute_timing_knn = knn(X_train, y_train, X_test, y_test, n_neighbors)

  if return_score_clf:
    return clf_score


# get list of all classes
data_all_classes = []
for i in range(10):
  data_all_classes.append(torch.from_numpy(MNIST_dataset_classe_all[i]))

# get MNIST test parameters
dataiter = iter(test_loader)
images, labels = dataiter.next()
data_MNIST_test = images.reshape((1, 10000, 1, 28, 28))

# Parameters
data_all_classes = data_all_classes
data = data_MNIST_test
nearest_centroid_score = True
knn_score = False
n_neighbors = 1 

model_list = [model_no_binary_stride, model_no_binary_maxpooling, model_binary_stride, model_binary_maxpooling, model_mixt_stride,  model_mixt_maxpooling
]
name_model = ['model_no_binary_stride', 'model_no_binary_maxpooling', 'model_binary_stride', 'model_binary_maxpooling', 'model_mixt_stride',  'model_mixt_maxpooling'
]


for k, model in enumerate(model_list):
  print('model: {}'.format(name_model[k]))
  model = model
  model_name = name_model[k]
  scores = {}
  list_layers = list(dict(model.named_children()).keys())
  if 'mixt' in model_name:
    mixt_model = True
  else:
    mixt_model = False

  for i in range(len(list_layers)):
    # get index layer by name:
    name_layer = list_layers[i]
    if (('batch' in name_layer) and not('max' in name_model[k])) or ('max' in name_layer):
      index_layer = list(dict(model.named_children()).keys()).index(name_layer)
      print('Layer: {} (index: {})'.format(name_layer, index_layer))
      
      # compute all classes average representation with maxpooling_size:
      if mixt_model:
        features_all_classes_no_binary, features_all_classes_binary, fm_shape = compute_features_all_classes(model, index_layer, data_all_classes, mixt_model=True)
      else:
        features_all_classes, fm_shape =  compute_features_all_classes(model, index_layer, data_all_classes)
      maxpooling_size_list = range(1, fm_shape+1)

      scores[name_layer] = []
      for j in maxpooling_size_list:
        print('maxpooling size: {}/{}'.format(j, fm_shape))
        maxpooling_size = j
        # get average_representation_all_classes:
        if mixt_model:
          features_all_classes_no_binary, features_all_classes_binary, _ = compute_features_all_classes(model, index_layer, data_all_classes, mixt_model=True)
          average_representation_all_classes_no_binary = compute_average_representation_all_classes(features_all_classes_no_binary, maxpooling_size)
          average_representation_all_classes_binary = compute_average_representation_all_classes(features_all_classes_binary, maxpooling_size)
          average_representation_all_classes_concatenate = torch.cat((average_representation_all_classes_no_binary, average_representation_all_classes_binary), 1)
          average_representation_all_classes = [average_representation_all_classes_no_binary, average_representation_all_classes_binary, average_representation_all_classes_concatenate]
          score_clf = compute_score_onn(model, average_representation_all_classes, data, index_layer, maxpooling_size, n_neighbors, nearest_centroid_score, knn_score, return_score_clf=True, mixt_model=True)
          scores[name_layer].append(score_clf)
        else:
          features_all_classes, _ =  compute_features_all_classes(model, index_layer, data_all_classes)
          average_representation_all_classes = compute_average_representation_all_classes(features_all_classes, maxpooling_size)
          score_clf = compute_score_onn(model, average_representation_all_classes, data, index_layer, maxpooling_size, n_neighbors, nearest_centroid_score, knn_score, return_score_clf=True)
          scores[name_layer].append(score_clf)
    else:
      continue

  # write python dict to a file
  path_save = 'results/MNIST_results/scores_onn/' + model_name + '_scores_clf' + '.pkl'
  output = open(path_save, 'wb')
  pickle.dump(scores, output)
  output.close()

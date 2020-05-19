In this repository we want study the effect of binary activations in convolutional layer.

We study these binary activations with two datasets: [Part1: MNIST](#part1-mnist-with-binary-activations), [Part2: Omniglot Classification](#part2-omniglot-classification-with-binary-activations) and [Part3: Omniglot Few shot](#part3-omniglot-few-shot-with-binary-activations).

This repository uses Pytorch library.

# Introduction: train discrete variables

To train a neural network with discrete variables, we can use two methods: REINFORCE (E (Williams, 1992; Mnih & Gregor,2014) and the straight-through estimator (Hinton, 2012; Bengio et al., 2013).

## Slope Annealing explicaion:
Extract from : ["HIERARCHICAL MULTISCALE RECURRENT NEURAL NETWORKS", Junyoung Chung, Sungjin Ahn & Yoshua Bengio (Mar 2017).](https://arxiv.org/pdf/1609.01704.pdf) : [2]

" Training neural networks with discrete variables requires more efforts since the standard backpropagation is no longer applicable due to the non-differentiability. Among a few methods for training a neural network with discrete variables such as the REINFORCE (Williams, 1992; Mnih & Gregor,2014) and the straight-through estimator (Hinton, 2012; Bengio et al., 2013). [...]
The straight-through estimator is a biased estimator because the non-differentiable function used in the forward pass (i.e., the step function in our case) is replaced by a differentiable function during the backward pass (i.e., the hard sigmoid function in our case). The straight-through estimator, however, is much simpler and often works more efficiently in practice than other unbiased but high-variance estimators such as the REINFORCE. The straight-through estimator has also been used in Courbariaux et al. (2016) and Vezhnevets et al. (2016).

The Slope Annealing Trick. In our experiment, we use the slope annealing trick to reduce the bias of the straight-through estimator. The idea is to reduce the discrepancy between the two functions used during the forward pass and the backward pass. That is, by gradually increasing the slope a of the hard sigmoid function, we make the hard sigmoid be close to the step function. Note that starting with a high slope value from the beginning can make the training difficult while it is more applicable later when the model parameters become more stable. In our experiments, starting from slope a = 1, we slowly increase the slope until it reaches a threshold with an appropriate scheduling. "


# PART1: MNIST with binary activations:
Most of the code in this section comes from this repository: [Github: Wizaron/binary-stochastic-neurons](https://github.com/Wizaron/binary-stochastic-neurons). [1]

In this part, we present results obtained with a simple 2 conv layer CNN. 

## Dataset:
The MNIST database of handwritten digits, available from this [link](http://yann.lecun.com/exdb/mnist/), has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. [5]

## Open Binary MNIST notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RVtgS6NOH5D5ssa19qDt35DYnjPxJCOX)

## Results on MNIST:

### First mixt architecture: 

CNN with two binary convolutionel layer and two no binary convolutional layer then concatenate the both.

### Loss/ACC: with 10 epochs.
|               Models: 2 conv layers (29k parameters)                  	|      Loss      	|  Accuracy (%)  	|
|:---------------------------------------------------------------------:	|:--------------:	|:--------------:	|
| No binary model with stride=2                                             |     **0.06**     	|      **98.07**    |
| No binary model with maxpooling=2                                         |     0.04      	|      98.59        |
|:---------------------------------------------------------------------:	|:--------------:	|:--------------:	|
| Mixt model with stride=2                                                 	|     **0.08**     	|      **97.49**    |
| Mixt model with maxpooling=2                                            	|     **0.05**     	|      **98.36**    |
|:---------------------------------------------------------------------:	|:--------------:	|:--------------:	|
| Stochastic binary model in the first conv layer with ST with stride=2     |     0.11       	|      96.71        |
| Stochastic binary model in the first conv layer with ST with maxpooling=2 |     **0.08**     	|      **97.57**    |

# PART2: Omniglot Classification with binary activations:

## Dataset:
Downlad from [Omniglot data set for one-shot learning](https://github.com/brendenlake/omniglot).

The Omniglot data set is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people. Each image is paired with stroke data, a sequences of [x,y,t] coordinates with time (t) in milliseconds. [6]

## Open Binary Omniglot notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z7ebfh8VWKfgkecpRzY7iiM6NT2ElWBL)

## Results on Omniglot classification with data train (80% train, 10% validation and 10% test):
### Loss/ACC: with 10 epochs.
|               Models: 4 conv layers          	                         	|  Accuracy (%)  	|
|:------------------------------------------------------------------------:	|:----------------:	|
| No binary model with stride=2                                             |     94.81     	|
| No binary model with maxpooling=2                                         |    **97.20**     	|
|:------------------------------------------------------------------------:	|:----------------:	|
| Mixt model with stride=2                                                 	|      93.62    	|
| Mixt model with maxpooling=2                                            	|    **96.06**     	|
|:------------------------------------------------------------------------:	|:----------------:	|
| Stochastic binary model in the first conv layer with ST with stride=2     |      94.66.    	|
| Stochastic binary model in the first conv layer with ST with maxpooling=2 |    **96.89**    	|



# PART3: Omniglot Few Shot Learning with binary activations:

Most of the code in this section comes from this repository: [Github: oscarknagg/few-shot](https://github.com/oscarknagg/few-shot). [3]

In this part, we present results obtained with [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) (Vinyals et al). [4]


## Open binary few shot Omniglot notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GW_yLVN62nRDgQrPAUkeM1b7yGSGSWg5)

## Results on Omniglot few shot learning:
### ACC: with this repository with 10 epochs.

|               Models: matching Network (MN) [4]              	|     Accuracy (%)   	|
|:-----------------------------------------------------------------:	|:--------------:	|
| k-way                                                  	|     5     	|
| n-shot                                                  	|    1    	|
|:-----------------------------------------------------------------:	|:--------------:	|
| No binary MN                                                  	|    **84.4**    	|
|:-----------------------------------------------------------------:	|:--------------:	|
| binary MN: first conv           	|     **79.6**     	|
| binary MN: second conv           	|     **79.6**     	|
| binary MN: third conv           	|     64.8     	|
| binary MN: fourth conv           	|     53.6     	|


# References: 
* [1]: [Github: Wizaron/binary-stochastic-neurons](https://github.com/Wizaron/binary-stochastic-neurons).
* [2]: ['HIERARCHICAL MULTISCALE RECURRENT NEURAL NETWORKS', Junyoung Chung, Sungjin Ahn & Yoshua Bengio (Mar 2017)](https://arxiv.org/pdf/1609.01704.pdf).
* [3]: [Github: oscarknagg/few-shot](https://github.com/oscarknagg/few-shot)
* [4]: ['Matching Networks for One Shot Learning', Vinyals et al (Dec 2017)](https://arxiv.org/pdf/1606.04080.pdf).
* [5]: [The MNIST Database](http://yann.lecun.com/exdb/mnist/).
* [6]: ['Omniglot data set for one-shot learning'](https://github.com/brendenlake/omniglot).
* [7]: ['Flashtorch '](https://github.com/MisaOgura/flashtorch).

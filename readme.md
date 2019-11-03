﻿### This repository contains PyTorch implementation of sparse autoencoder and it's application for image denosing and reconstruction.

Autoencoder (AE) is an unsupervised deep learning algorithm, capable of extracting useful features from data. To do so, the model tries to learn an approximation to identity function, setting the labels equal to input. Although learning an identity function may seem like an easy task, placing some constrains on a model makes it discover essential features for reconstructing input data. Typically, these constraints are imposed on the middle layer of AE model, and consist in limiting the number of neurons.

Autoencoder architecture.
![SAE](/images/SAE.jpeg)

In the image above, AE is applied to image from MNIST dataset with size 28*28 pixels. Passing it through middle layer (also called latent space), which has 10 neurons, network is forced to learn a lower dimension representation of the image, thus learning to reconstruct a 784-dimensional data from 10-dimensional space.

This way, AEs can be used as dimension reduction algorithm similar to PCA. Another major application of AEs is data denoising. Previously, AEs was also used for such tasks as pre-training of deep networks.

Restrictions on latent representation can be imposed not only by limiting number of neurons, but also by adding some term in the loss function. By imposing sparsity constraint, the latent layer can have even more neurons than number of input dimensions. And that type of AE (sparse autoencoder, SAE) will still be able to discover interesting features in the input data.

The sparsity constraint is penalizing activations of neurons in such way, that only few of them can be active at the same time. By "active" here means that activation of this particular neuron is close to 1, while inactive neurons activate close to 0. Most of the time neurons should be inactive. So with only few hidden units active for some input data and ability to reconstruct input one can say that model has learned some usefull features from data and not overfitting.

One way to achive that is by adding to the loss function a Kullback-Leibler divergence between Bernoulli distribution whith mean <img src="/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/> and distribution of latent layer activations:

<img src="/tex/77cfa7d35f1f6e57bffec24519aaf628.svg?invert_in_darkmode&sanitize=true" align=middle width=274.40292165pt height=33.20539859999999pt/>

where <img src="/tex/2ece8916c80529609c5cc5d5b4e259f4.svg?invert_in_darkmode&sanitize=true" align=middle width=9.728951099999989pt height=22.831056599999986pt/> is the mean of distribution of latent neurons activations over training data. Setting <img src="/tex/6dec54c48a0438a5fcde6053bdb9d712.svg?invert_in_darkmode&sanitize=true" align=middle width=8.49888434999999pt height=14.15524440000002pt/> to small value will force hidden neurons activations be mostly close to 0. Thus, this is a way of regularizing activations of neurons and make them data-dependend, where different neurons "fire" from different input samples.




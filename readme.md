### This repository contains PyTorch implementation of sparse autoencoder and it's application for image denosing and reconstruction.

Autoencoder (AE) is an unsupervised deep learning algorithm, capable of extracting useful features from data. To do so, the model tries to learn an approximation to identity function, setting the labels equal to input. Although learning an identity function may seem like an easy task, placing some constrains on a model makes it discover essential features for reconstructing input data. Typically, these constraints are imposed on the middle layer of AE model, and consist in limiting the number of neurons.

Autoencoder architecture.
![SAE](/images/SAE.jpeg)

In the image above, AE is applied to image from MNIST dataset with size 28*28 pixels. Passing it through middle layer (also called latent space), which has 10 neurons, network is forced to learn a lower dimension representation of the image, thus learning to reconstruct a 784-dimensional data from 10-dimensional space.

This way, AEs can be used as dimension reduction algorithm similar to PCA. Another major application of AEs is data denoising. Previously, AEs was also used for such tasks as pre-training of deep networks.

Restrictions on latent representation can be imposed not only by limiting number of neurons, but also by adding some term in the loss function. By imposing sparsity constraint, the latent layer can have even more neurons than number of input dimensions. And that type of AE (sparse autoencoder, SAE) will still be able to discover interesting features in the input data.

The sparsity constraint is penalizing activations of neurons in such way, that only few of them can be active at the same time. By "active" here means that activation of this particular neuron is close to 1, while inactive neurons activate close to 0. Most of the time neurons should be inactive. So with only few hidden units active for some input data and ability to reconstruct input one can say that model has learned some usefull features from data and not overfitting.

One way to achive that is by adding to the loss function a Kullback-Leibler divergence between Bernoulli distribution whith mean ![rho](http://chart.apis.google.com/chart?cht=tx&chl=_\rho) and distribution of latent layer activations:

$D_{KL}(\rho||\hat{\rho}) = \rho\log{\frac{\rho}{\hat{\rho}}} + (1-\rho)\log{\frac{(1-\rho)}{(1-\hat{\rho})}}$,

![DKL](http://chart.apis.google.com/chart?cht=tx&chl=D_{KL}(\rho||\hat{\rho})=\rho\log{\frac{\rho}{\hat{\rho}}}{+}(1-\rho)\log{\frac{(1-\rho)}{(1-\hat{\rho})}})

where ![rhohat](http://chart.apis.google.com/chart?cht=tx&chl=\hat{_\rho}) is the mean of distribution of latent neurons activations over training data. Setting $\rho$ to small value will force hidden neurons activations be mostly close to 0. Thus, this is a way of regularizing activations of neurons and make them data-dependend, where different neurons "fire" from different input samples.




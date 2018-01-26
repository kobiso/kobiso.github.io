---
title: "Autoencoder"
categories:
  - Research
tags:
  - autoencoder
  - PCA
header:
  teaser: /assets/images/autoencoder/structure.png
  overlay_image: /assets/images/autoencoder/structure.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

**Autoencoder** is an artificial neural network used for unsupervised learning of efficient codings.
The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.
Recently, the autoencoder concept has become more widely used for learning generative models of data.

{% include toc title="Table of Contents" icon="file-text" %}

# Autoencoder
- For the intuitive understanding, **autoencoder** compresses (learns) the input and then reconstruct (generates) of it.
  - It is **unsupervised learning** model which does not need label.  

$$
\text{Input'} = \text{Decoder(Encoder(Input))}
$${: .text-cetner}

## Structure
![Autoencoder]({{ site.url }}{{ site.baseurl }}/assets/images/autoencoder/structure.png){: .align-center}

- Basic form of an autoencoder is a feedforward neural network which has an input layer, an output layer and one or more hidden layers.
  - Output layer has the same number of nodes as the input layer for reconstructing its own inputs.
  
- An autoencoder consists of two parts, the **encoder** $$\phi$$ and the **decoder** $$\psi$$
  - They will be chosen by minimizing the distance between $$X$$ and $$X'$$.

$$
\phi:\mathcal{X} \rightarrow \mathcal{F} \\
\psi:\mathcal{F} \rightarrow \mathcal{X} \\
\phi,\psi = \underset{\phi,\psi}{\operatorname{arg\,min}}\, \|X-(\psi \circ \phi) X\|^2
$${: .text-center}

- The encoder stage of an autoencoder take the input $$\mathbf{x} \in \mathbb{R}^d = \mathcal{X}$$ and maps it to $$\mathbf{z} \in \mathbb{R}^p = \mathcal{F}$$.
  - The $$z$$ is referred to as *code*, *latent representation*.
  - $$W$$ is a weight, $$b$$ is a bias vector, $$\sigma$$ is activation function for non-linearity.
  - The decoder stage of the autoencoder maps $$z$$ to the reconstruction $$x'$$ of the same shape as $$x$$.

$$
\mathbf{z} = \sigma(\mathbf{Wx}+\mathbf{b})
\mathbf{x'} = \sigma'(\mathbf{W'z}+\mathbf{b'})
$${: .text-center}

- **Loss function**: Autoencoder is trained to minimise reconstruction error (squared error) where $$x$$ is usually averaged over some input training set.

$$
\mathcal{L}(\mathbf{x},\mathbf{x'})=\|\mathbf{x}-\mathbf{x'}\|^2=\|\mathbf{x}-\sigma'(\mathbf{W'}(\sigma(\mathbf{Wx}+\mathbf{b}))+\mathbf{b'})\|^2
$${: .text-center}

## Autoencoder for PCA
The first paper suggested autoencoder [(Baldi, P. and Hornik, K. 1989)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.1839&rep=rep1&type=pdf) proposed the implementation of PCA using backpropagation of neural network

### PCA
- PCA takes N-dimentional data and finds the M orthogonal directions in which the data have the most variance.
  - These M principal directions form a lower-dimensional subspace.
  - We can represent an N-dimensional datapoint by its projections onto the M principal directions.
  - This loses all information about where the datapoint is located in the remaining orthogonal directions.
- We reconstruct by using the mean value (over all the data) on the N-M directions that are not represented.
  - The reconstruction error (squared distance) is the sum over all these unrepresented directions of the squared differences of the datapoint from the mean.
  
![PCA]({{ site.url }}{{ site.baseurl }}/assets/images/autoencoder/pca.png){: .align-center}

### PCA Using Autoencoder
- If the hidden and output layers are linear, it will learn hidden units that are a linear function of the data and minimize the squared reconstruction error which is exactly what PCA does.
  - Autoencoder will be trained to minimise the reconstruction error which tries to make input $$x$$ and output $$x'$$ the same.
  - The code $$z$$ with $$M$$ hidden unit will be compressed representation of input $$N$$
  - It is usually inefficient than PCA, which can be improved with big data set. 
- The M hidden units will span the same space as the first M components found by PCA.
  - Their weight vectors may not be orthogonal.
  - They will tend to have equal variances.
- with non-linear layers before and after the code, it should be possible to efficiently represent data that lies on or near a non-linear manifold.
  - The encoder converts coordinates in the input space to coordinates on the manifold.
  - The decoder does the inverse mapping  

# References
- Wikipedia: Autoencoder [[Link](https://en.wikipedia.org/wiki/Autoencoder)]
- Youtube: From PCA to autoencoders [Neural Networks for Machine Learning] [[Link](https://www.youtube.com/watch?v=hbU7nbVDzGE)]
- Paper: Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima [[Link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.1839&rep=rep1&type=pdf)]
---
title: "Feed-Forward Neural Network (FFNN)"
categories:
  - Research
tags:
  - ffnn
header:
  teaser: /assets/images/ffnn/ffnn.png
  overlay_image: /assets/images/ffnn/ffnn.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

{% include toc title="Table of Contents" icon="file-text" %}

# Feed-Forward Neural Network (FFNN)
  
> A **feed-forward neural network** is an artificial neural network wherein connections  between the units do not form a cycle.
\- [Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network) 

FFNN is often called *multilayer perceptrons (MLPs)* and *deep feed-forward network* when it includes many hidden layers.
It consists of an input layer, one or several hidden layers, and an output layer when every layer has multiple neurons (units).
Each connection between neurons is associated with *weight* and *bias*.

![Example architecture of LRCN]({{ site.url }}{{ site.baseurl }}/assets/images/ffnn/ffnn.png){: .align-center}

## Forward Propagation
Consider a FFNN with $$L$$ hidden layers, $$l\in\{1,...,L\}$$ is index of the hidden layers.
The forward propagation process of FFNN can be described as below.

$$
z^{(l+1)}_{i} = \mathbf{w}^{(l+1)}_{i}\mathbf{y}^{(l)} + b^{(l+1)}_{i}, \\
y^{(l+1)}_{i} = f(z^{(l+1)}_{i}), 
$${: .text-center}

- $$\mathbf{z}^{(l)}$$: the vector of inputs into layer $$l$$
- $$\mathbf{y}^{(l)}$$: the vector of outputs from layer $$l$$
  - $$\mathbf{y}^{(1)}$$ equals to $$\mathbf{x}$$ as the input
- $$i$$: any hidden unit
- $$\mathbf{W}^{(l)}$$: a weight matrix in the layer $$l$$
- $$\mathbf{b}^{(l)}$$: a bias vector in the layer $$l$$
- $$f(\cdot)$$: an [activation function](/research/research-activation-functions/) such as a sigmoid, hyperbolic tangent, rectified linear unit, or softmax function.

![Example architecture of LRCN]({{ site.url }}{{ site.baseurl }}/assets/images/ffnn/ffnn_equ.png){: .align-center}

# References
- FFNN in Wikipedia [[Link](https://en.wikipedia.org/wiki/Feedforward_neural_network)]
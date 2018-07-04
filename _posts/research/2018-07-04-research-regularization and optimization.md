---
title: "Regularization and Optimization"
categories:
  - Research
tags:
  - paper skimming
  - image retrieval
header:
  teaser: /assets/images/regularization and optimization/penalizing.png
  overlay_image: /assets/images/regularization and optimization/penalizing.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post is a summary and paper skimming on regularization and optimization.
So, this post will be keep updating by the time.

{% include toc title="Table of Contents" icon="file-text" %}

# Paper List

## Regularization
- [Regularizing neural networks by penalizing confident output distributions]({{ site.url }}{{ site.baseurl }}/research/research-regularization-and-optimization/#regularizing-neural-networks-by-penalizing-confident-output-distributions), ICLR2017, Google, Geoffrey Hinton
  - [Paper](https://arxiv.org/pdf/1701.06548.pdf)

## Optimization

# Regularizing neural networks by penalizing confident output distributions
- Conference: ICLR2017

## Summary
 
- **Research Objective**
  - To suggest the wide applicable regularizers
  
- **Proposed Solution**
  - Regularizing neural networks by penalizing low entropy output distributions
  - Penalizing low entropy output distributions acts as a strong regularizer in supervised learning.
  - Connect a maximum entropy based confidence penalty to label smoothing through the direction of the KL divergence.
    - When the prior label distribution is uniform, label smoothing is equivalent to adding the KL divergence between the uniform distribution $$u$$ and the network's predicted distribution $$p_\theta$$ to the negative log-likelihood.
    - By reversing the direction of the KL divergence in equation (1), $$D_{KL}(u \parallel p_\theta(y \mid x))$$, it recovers the confidence penalty.

$$
\mathcal{L}(\theta)=-\sum \log p_\theta (y\mid x)-D_{KL}(u \parallel p_\theta(y \mid x)) \cdots (1)
$$

![Comparision]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/penalizing.png){: .align-center}{:height="100%" width="100%"}
*Figure: Distribution of the magnitude of softmax probabilities on the MNIST validation set. A fully-connected, 2-layer, 1024-unit neural network was trained with dropout (left), label smoothing (center), and the confidence penalty (right). Dropout leads to a softmax distribution where probabilities are either 0 or 1. By contrast, both label smoothing and the confidence penalty lead to smoother output distributions, which results in better generalization.*
{: .text-center}

- **Contribution**
  - Both label smoothing and the confidence penalty improve state-of-the-art models across benchmarks without modifying existing hyperparameters

![Result]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/penalizing2.png){: .align-center}{:height="100%" width="100%"}
*Figure: Test error (%) for permutation-invariant MNIST.*
{: .text-center}

## References
- Paper: [Regularizing neural networks by penalizing confident output distributions](https://arxiv.org/pdf/1701.06548.pdf)
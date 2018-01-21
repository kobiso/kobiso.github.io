---
title: "Generative Adversarial Networks (GAN)"
categories:
  - Research
tags:
  - GAN
header:
  teaser: /assets/images/gan/intuition.png
  overlay_image: /assets/images/gan/intuition.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

**Generative Adversarial Networks (GAN)** is a framework for estimating generative models via an adversarial process by training two models simultaneously.
A generative model *G* that captures the data distribution, and a discriminative model *D* that estimates the probability that a sample came from the training data rather than G.
It was proposed and presented in [Advances in Neural Information Processing Systems (NIPS) 2014](https://papers.nips.cc/paper/5423-generative-adversarial-nets).

{% include toc title="Table of Contents" icon="file-text" %}

# Introduction
- Taxonomy of **Machine Learning**
![ML]({{ site.url }}{{ site.baseurl }}/assets/images/gan/ml.png){: .align-center}
  - **Supervised learning**: The discriminative model learns how to classify input to its class.
  ![Discriminative]({{ site.url }}{{ site.baseurl }}/assets/images/gan/discriminative.png){: .align-center}
  - **Unsupervised learning**: The generative model learns the distribution of training data.
  ![Generative]({{ site.url }}{{ site.baseurl }}/assets/images/gan/generative.png){: .align-center}
    - More challenging than supervised learning because there is no label or curriculum which leads self learning
    - NN solutions: Boltzmann machine, Auto-encoder, Variation Inference, GAN 
  
- The goal of the generative model is to find a $$p_{model}(x)$$ that approximates $$p_{data}(x)$$ well.
![Goal]({{ site.url }}{{ site.baseurl }}/assets/images/gan/goal.png){: .align-center}

# Generative Adversarial Nets (GAN)
## Intuition of GAN
- The **discriminator** *D* should classify a real image as real ($$D(x)$$ close to 1) and a fake image as fake ($$D(G(z))$$ close to 0).
- The **generator** *G* should create an image that is indistinguishable from real to deceive the discriminator ($$D(G(z))$$ close to 1).
![Intuition]({{ site.url }}{{ site.baseurl }}/assets/images/gan/intuition.png){: .align-center}
{: .full}

![Explanation]({{ site.url }}{{ site.baseurl }}/assets/images/gan/explanation.png){: .align-center}
{: .full}

## Objective Function

- **Objective function** of GAN is minimax game of two-player *G* and *D*.

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{data}~(x)}[log D(x)] + \mathbb{E}_{z\sim p_z(z)}[log(1-D(G(z)))]
$${: .text-center}

- For the **discriminator** *D* should maximize $$V(D,G)$$
  - Sample $$x$$ from real data distribution for $$p_{data}(x)$$
  - Sample latent code $$z$$ from Gaussian distribution for $$p_z(z)$$
  - $$V(D,G)$$ is maximum when $$D(x)=1$$ and $$D(G(z))=0$$
  
- For the **generator** *G* should minimize $$V(D,G)$$
  - *G* is independent of $$\mathbb{E}_{x\sim p_{data}~(x)}[log D(x)]$$
  - $$V(D,G)$$ is minimum when $$D(G(z))=1$$
  
- **Saturating problem**
  - In practice, early in learning, when *G* is poor, *D* can reject samples with high confidence because they are clearly different from the training data.
  - In this case, the gradient is relatively small at $$D(G(z))=0$$ which makes $$\log (1-D(G(z)))$$ saturates.
  ![Saturate]({{ site.url }}{{ site.baseurl }}/assets/images/gan/saturate.png){: .align-center}
  - Rather than training *G* to minimize $$\log (1-D(G(z)))$$, we can tran *G* to maximize $$\log D(G(z))$$.
  - This objective function results in the same fixed point of the dynamics of *G* and *D* but provides much stronger gradients early in learning.
  ![Non-Saturate]({{ site.url }}{{ site.baseurl }}/assets/images/gan/non-saturate.png){: .align-center}
  
- **Why does GANs work?**
  - Because it actually minimizes the distance between the real data distribution $$p_{data}$$ and the model distribution $$p_g$$.
    - [**Jensen-Shannon divergence (JSD)**](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) is a method of measuring the similarity between two probability distributions based on the **Kullback-Leibler divergence (KL)**.
  
![Why]({{ site.url }}{{ site.baseurl }}/assets/images/gan/why.png){: .align-center}
{: .full}

## Implementation Example
![Framework]({{ site.url }}{{ site.baseurl }}/assets/images/gan/framework.png){: .align-center}

# Variants of GAN
## Deep Convolutional GAN (DCGAN), 2015
- **DCGAN** used convolution for discriminator and deconvolution for generator.
  - It is stable to train in most settings compared to GANs.
  - DCGAN used the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
  - Specific filters of DCGAN have learned to draw specific objects.
  - The generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.  

![DCGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/dcgan.png){: .align-center}
{: .full}

- **Latent vector arithmetic**
  - They showed consistent and stable generations that semantically obeyed the linear arithmetic including object manipulation and face pose.
  
![Arithmetic]({{ site.url }}{{ site.baseurl }}/assets/images/gan/dcgan2.png){: .align-center}

## Least Squares GAN (LSGAN), 2016
- **LSGAN** adopt the **least squares loss function** for the discriminator instead of cross entropy loss function of GAN.
  - Since, cross entropy loss function may lead to the **vanishing gradient problem**.
  - LSGANs are able to generate higher quality images than regular GANs.
  - LSGANs performs more stable during the learning process.

![LSGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/lsgan.png){: .align-center}
{: .full}

## Semi-Supervised GAN (SGAN), 2016
- **SGAN** extend GANs that allows them to learn a generative model and a classifier simultaneously.
  - SGAN improves classification performance on restricted data sets over a baseline classifier with no generative component.
  - SGAN can significantly improve the quality of the generated samples and reduce training times for the generator. 

![SGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/sgan.png){: .align-center}
{: .full}

## Auxiliary Classifier GAN (ACGAN), 2016
- **ACGAN** is added more structure to the GAN latent space along with a specialized cost function results in higher quality samples.

![ACGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/acgan.png){: .align-center}
{: .full}

# Extensions of GAN
## CycleGAN: Unpaired Image-to-Image Translation
- **CycleGAN** presents a GAN model that transfer an image from a source domain A to a target domain B in the absence of paired examples.
  - The generator $$G_{AB}$$ should generates a horse from the zebra to deceive the discriminator $$D_B$$.
  - $$G_{BA}$$ generates a reconstructed image of domain A which makes the shape to be maintained when $$G_{AB}$$ generates a horse image from the zebra.

![CycleGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/cyclegan.png){: .align-center}
{: .full}

- **Result**
![CycleGAN Result]({{ site.url }}{{ site.baseurl }}/assets/images/gan/cyclegan2.png){: .align-center}

## StackGAN: Text to Photo-realistic Image Synthesis
- **StackGAN** generate $$256 \times 256$$ photo-realistic images conditioned on text descriptions.

![StackGAN]({{ site.url }}{{ site.baseurl }}/assets/images/gan/stackgan.png){: .align-center}
{: .full}

- **Result**
![StackGAN Result]({{ site.url }}{{ site.baseurl }}/assets/images/gan/stackgan2.png){: .align-center}

## Latest work
- **Visual Attribute Transfer**
  - Jing Liao et al. Visual Attribute Transfer through Deep Image Analogy, 2017
  
![Visual]({{ site.url }}{{ site.baseurl }}/assets/images/gan/visual.png){: .align-center}

- **User-Interactive Image Colorization**
  - Richard Zhang et al. Real-Time User-Guided Image Colorization with Learned Deep Prioirs, 2017
  
![Colorization]({{ site.url }}{{ site.baseurl }}/assets/images/gan/colorization.png){: .align-center}

# References
- Paper: Generative Adversarial Nets [[Link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)]
- Paper: Unsupervised representation learning with deep convolutional generative adversarial networks [[Link](https://arxiv.org/pdf/1511.06434.pdf)]
- Paper: Least Squares Generative Adversarial Networks [[Link](https://arxiv.org/abs/1611.04076)]
- Paper: Semi-Supervised Learning with Generative Adversarial Networks [[Link](https://arxiv.org/abs/1606.01583)]
- Paper: Conditional Image Synthesis With Auxiliary Classifier GANs [[Link](https://arxiv.org/abs/1610.09585)]
- Paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[Link](https://arxiv.org/abs/1703.10593)]
- Paper: StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks [[Link](https://arxiv.org/abs/1612.03242)]
- PPT: Generative Adversarial Nets by Yunjey Choi [[Link](https://www.slideshare.net/NaverEngineering/1-gangenerative-adversarial-network)]
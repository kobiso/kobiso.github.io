---
title: "Status of dropout technique usage in famous deep networks"
categories:
  - Research
tags:
  - Dropout
  - Deep network
header:
  teaser: /assets/images/dropout.jpeg
---

## Dropout technique

> Dropout is one of the simplest and the most powerful regularization techniques.
It prevents units from complex co-adapting by randomly dropping units from the network. [[N. Srivastava et al., 2014](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)]
Below table is a list of famous deep networks which use dropout techniques.

## Status of dropout technique usage in famous deep networks

| Model | Dropout layers | Remark |
|:--------|:-------:|--------:|
| AlexNet [[Alex Krizhevsky et al., 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)]   | Used in two fully-connected layers   | Won the 2012 ILSVRC (ImageNet Large-Scale Visual Recognition Challenge)   |
| ZFNet [[Matthew D. Zeiler et al., 2013](https://arxiv.org/pdf/1311.2901v3.pdf)]   | Used in two fully-connected layers   | Won the 2013 ILSVRC   |
| VGG Net [[Karen Simonyan et al., 2014](https://arxiv.org/pdf/1409.1556v6.pdf)]   | Used in two fully-connected layers   | Best utilized with simple and deep CNN   |
| GoogLeNet [[Christian Szegedy et al., 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)]   | Used in one fully-connected layer   | Won the 2014 ILSVRC   |
| Generative Adversarial Networks [[Ian J. Goodfellow et al., 2014](https://arxiv.org/pdf/1406.2661v1.pdf)]   | Applied in training the discriminator net   | Various usage such as feature extraction, generating artificial images   |
| Generating Image Descriptions [[Adrej Karpathy et al., 2014](https://arxiv.org/pdf/1412.2306v2.pdf)]   | Used in all layers except in the recurrent layers   | Combination of CNNs and RNNs   |
| Spatial Transformer Networks [[Max Jaderberg et al., 2015](https://arxiv.org/pdf/1506.02025.pdf)]   | Used in all layers except the first convolutional layer   | Introduce of a Spatial Transformer module   |
{: rules="groups"}

## Notices

The list will keep updated
{: .notice}

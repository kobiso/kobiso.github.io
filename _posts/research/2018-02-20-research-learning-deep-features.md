---
title: "Learning Deep Features for Discriminative Localization"
categories:
  - Research
tags:
  - global average pooling
  - GAP
  - class activation map
  - CAM
header:
  teaser: /assets/images/learning deep features/class activation map.png
  overlay_image: /assets/images/learning deep features/class activation map.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"Learning Deep Features for Discriminative Localization" proposed a method to enable the convolutional neural network to have localization ability despite being trained on image-level labels.
It was presented in Conference on Computer Vision and Pattern Recognition (CVPR) 2016 by B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba.

{% include toc title="Table of Contents" icon="file-text" %}

# Introduction
- **Weakly-supervised object localization**
  - [Oquab et al.](http://www.di.ens.fr/~josef/publications/Oquab15.pdf) proposed object localization with *global max pooling (GMP)*.
  - Instead of GAP, this paper proposed to use *global average pooling (GAP)* since the loss for average pooling benefits when the network identifies all discriminative regions for an object as compared to max pooling.
  - The authors of this paper claimed that, while global average pooling is not a novel technique that this paper propose, the observation that it can be applied for accurate discriminative localization is novel.
   
# Class Activation Mapping
- Below figure show that the predicted class score is mapped back to the previous convolutional layer to generate the class activation maps (CAMs) which highlights the class-specific discriminative regions.

![Class Activation Mapping]({{ site.url }}{{ site.baseurl }}/assets/images/learning deep features/class activation map.png){: .align-center}
{:.full}

# References
- Paper: Learning Deep Features for Discriminative Localization [[Link](https://arxiv.org/abs/1512.04150)]
- Web: Implementation by the paper author [[Link](http://cnnlocalization.csail.mit.edu/)]
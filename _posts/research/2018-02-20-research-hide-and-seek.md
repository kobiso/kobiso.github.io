---
title: "Hide-and-Seek"
categories:
  - Research
tags:
  - hide-and-seek
  - object detection
  - weakly-supervised object
header:
  teaser: /assets/images/hide-and-seek/main idea.png
  overlay_image: /assets/images/hide-and-seek/main idea.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization" proposed a weakly-supervised framework to improve object localization in images and action localization in videos.
It was presented in International Conference on Computer Vision (ICCV) 2017 by Krishna Kumar Singh and Yong Jae Lee.

{% include toc title="Table of Contents" icon="file-text" %}

# Summary
- **Problem Statement**
  - Most existing weakly-supervised methods localize only the most discriminative parts of an object rather than all relevant parts.
  - This leads to suboptimal performance.
  
- **Research Objective**
  - To improve object localization in images and action localization in videos.

- **Solution Proposed: Hide-and-Seek**
  - The key idea is to hide patches in a training image randomly, forcing the network to seek other relevant parts when the most discriminative part is hidden.
  
- **Contribution**
  - Introduce the idea of Hide-and-Seek for weakly-supervised localization and produce state-of-the-art object localization results on the ILSVRC dataset
  - Demonstrate the generalizability of the approach on different networks and layers
  - Extend the idea to the relatively unexplored task of weakly-supervised temporal action localization  

# Introduction
- Previous weakly-supervised methods often fail to identify the entire extent of the object and instead localize only the most discriminative part

- **Main Idea**
  - Proposed method make changes to the input image.
  - The key idea is to *hide* patches from an image during training so that the model needs to *seek* the relevant object parts from what remains.
  - By randomly hiding different patches in each training epoch, the model sees different parts of the image
  and is forced to focus on multiple relevant parts of the object beyond just the most discriminative one.
  - This random hiding of patches will be only applied on training, not on testing. 

![Main Idea]({{ site.url }}{{ site.baseurl }}/assets/images/hide-and-seek/main idea.png){: .align-center}{:height="75%" width="75%"}

# Proposed Method
## Weakly-supervised Object Localization
- **Goal**: To learn an object localizer that can predict both the category label as well as the bounding box for the object-of-interest in a new test image $$I_test$$
  - Given a set of images $$I_{set}={I_1, I_2, ..., I_N}$$ where each image $$I$$ is labeled only with its category label.
  - In order to learn the object localizer, we train a CNN which simultaneously learns to localize the object while performing the image classification task.
  - Existing localizing method only focus on discriminative object parts, since they are sufficient for optimizing the classification task.




![Approach]({{ site.url }}{{ site.baseurl }}/assets/images/hide-and-seek/approach.png){: .align-center}
{: .full}

# References
- Paper: Hide-and-Seek: Forcing a Network to be Meticulous for Weakly-supervised Object and Action Localization [[Link](https://arxiv.org/abs/1704.04232)]
- Github: Hide-and-Seek (Implementation by the paper author) [[Link](https://github.com/kkanshul/Hide-and-Seek)]
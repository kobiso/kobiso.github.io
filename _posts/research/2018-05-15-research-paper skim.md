---
title: "Paper Skim"
categories:
  - Research
tags:
  - paper skim
header:
  teaser: /assets/images/paper skim/land mark.png
  overlay_image: /assets/images/paper skim/land mark.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post is a summary or paper skimming on various subject of deep learning.
So, this post will be keep updating by the time.

{% include toc title="Table of Contents" icon="file-text" %}

# Revisiting Dilated Convolution: A Simple Approach for Weakly- and Semi- Supervised semantic Segmentation
- Conference: CVPR2018
- Institute: UIUC, NUS, IBM, Tencent

## Summary
- **Problem Statement**
  - Time-consuming boudning box annotation is sidestepped in *weakly supervised learning*.
  - In this case, the supervised information is restricted to binary labels (object absence/presence) without their locations.
  
- **Research Objective**
  - To infer the object locations during weakly supervised learning
  
- **Proposed Solution**
  - Propose a multiple-instance learning approach that iteratively trains the detector and infers the object locations in the positive training images
  - Window refinement method
  
- **Contribution**
  - Multi-fold multiple instance learning procedure, which prevents training from prematurely locking onto erroneous object locations
  - Window refinement method improves the localization accuracy by incorporating an objectness prior.

![Algorithm]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/mil algo.png){: .align-center}{:height="80%" width="80%"}
*Figure: Multi-fold weakly supervised training*
{: .text-center}

## References
- Paper: [Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning](https://arxiv.org/pdf/1503.00949.pdf)

# Unsupervised Learning of Object Landmarks by Factorized Spatial embeddings
- Conference: ICCV2016
- Institute: University of Oxford

## Summary
- **Problem Statement**
  - Learning automatically the structure of object categories is an oppen problem in computer vision.
  
- **Research Objective**
  - To learn landmarks of objects with unsupervised approach
  
- **Proposed Solution**
  - Propose a unsupervised approach that can discover and learn landmarks in object categories, thus characterizing their structure.
  - Approach is based on factorizing image deformations, as induced by a viewpoint change or an object deformation, by learning a deep neural network that detects landmarks consistently with such visual effects.
  
- **Contribution**
  - Learned-landmarks establish meaningful correspondences between different object instances in a category without having to impose this requirement explicitly.
  - Proposed unsupervised landmarks are highly predictive of manually-annotated landmarks in face benchmark datasets, and can be used to regree these with a high degree of accuracy.

![Algorithm]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/land mark.png){: .align-center}{:height="80%" width="80%"}
*Figure: Proposed method that cna learn view point invariant landmarks without any supervision.*
{: .text-center}

## References
- Paper: [Unsupervised Learning of Object Landmarks by Factorized Spatial embeddings](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/thewlis17unsupervised.pdf)

<!--
# Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning, TPAMI16
- Conference: TPAMI2016
- Institute: 

## Summary
- **Problem Statement**
  - Time-consuming boudning box annotation is sidestepped in *weakly supervised learning*.
  - In this case, the supervised information is restricted to binary labels (object absence/presence) without their locations.
  
- **Research Objective**
  - To infer the object locations during weakly supervised learning
  
- **Proposed Solution**
  - Propose a multiple-instance learning approach that iteratively trains the detector and infers the object locations in the positive training images
  - Window refinement method
  
- **Contribution**
  - Multi-fold multiple instance learning procedure, which prevents training from prematurely locking onto erroneous object locations
  - Window refinement method improves the localization accuracy by incorporating an objectness prior.

![Algorithm]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/mil algo.png){: .align-center}{:height="80%" width="80%"}
*Figure: Multi-fold weakly supervised training*
{: .text-center}

## References
- Paper: Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning [[Link](https://arxiv.org/pdf/1503.00949.pdf)]
-->
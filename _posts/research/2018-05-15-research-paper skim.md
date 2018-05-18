---
title: "Paper Skim"
categories:
  - Research
tags:
  - paper skim
header:
  teaser: /assets/images/paper skim/mil algo.png
  overlay_image: /assets/images/paper skim/mil algo.png
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
- Paper: Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning [[Link](https://arxiv.org/pdf/1503.00949.pdf)]

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

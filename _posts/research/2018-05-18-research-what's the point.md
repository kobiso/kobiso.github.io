---
title: "What's the point: semantic segmentation with point supervision"
categories:
  - Research
tags:
  - semantic segmentation
  - weak supervision
  - data annotation
header:
  teaser: /assets/images/what's the point/segmentation method.png
  overlay_image: /assets/images/what's the point/segmentation method.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"What's the point: semantic segmentation with point supervision" proposes a way to incorporate point supervision along with a semantic segmentation.
This paper was presented in the European Conference on Computer Vision (ICCV) 2016 by Amy Bearman, Olga Russakovsky, Vittorio Ferrari, and Li Fei-Fei.

{% include toc title="Table of Contents" icon="file-text" %}

# Summary

- **Problem Statement**
  - Detailed per-pixel annotations enable training accurate models but are very time-consuming to obtain
  - Image-level class labels are an order of magnitude cheaper but result in less accurate models
  
- **Research Objective**
  - To take a natural step (point) from image-level annotation towards stronger supervision
  
- **Proposed Solution**
  - Annotators *point* to an object if one exists
  - Incorporate this point supervision along with a novel objectness potential in the training loss function of a CNN model.
  
- **Contribution**
  - Experimental results on the PASCAL VOC 2012 benchmark reveal that the combined effect of point-level supervision and objectness potential yields an improvement of 12.9% mIOU over image-level supervision
  - Models trained with point-level supervision are more accurate than models trained with image-level, squiggle-level or full supervision given a fixed annotation budget

# Semantic Segmentation Method
segmentation method.png

![Segmentation Method]({{ site.url }}{{ site.baseurl }}/assets/images/what's the point/segmentation method.png){: .align-center}{:height="100%" width="100%"}
*Figure:(Top): Overview of our semantic segmentation training framework. (Bottom): Different levels of training supervision*
{: .text-center}

# References
- Paper: [Extreme clicking for efficient object annotation, ECCV2016](https://arxiv.org/pdf/1708.02750.pdf)
- Reference paper: [Fully Convolutional Networks for Semantic Segmentation, CVPR 2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
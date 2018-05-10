---
title: "Training object class detectors with click supervision"
categories:
  - Research
tags:
  - click supervision
  - bounding box
header:
  teaser: /assets/images/click supervision/workflow.png
  overlay_image: /assets/images/click supervision/workflow.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"Training object class detectors with click supervision" proposes efficient way of annotating bounding boxes for object class detectors.
It was presented in the Conference on Computer Vision and Pattern Recognition (CVPR) 2017 by Jasper R. R. Uijlings and Vittorio Ferrari (Google AI Perception).
(This article is still on writing...)

{% include toc title="Table of Contents" icon="file-text" %}

# Summary

- **Problem Statement**
  - Manually drawing bounding boxes for training object class detectors is time consuming.
  
- **Proposed Solution**
  - Greatly reduce annotation time by proposing center-click annotation
    - Annotators click on the center of an imaginary bounding box
    - Incorporate these clicks into existing Multiple Instance Learning techniques for weakly supervised object localization    
  
- **Contribution**
  - Delivers high-quality detectors, performing substantially better than those produced by weakly supervised techniques, with a modest extra annotation effort
  - Perform in a range close to those trained from manually drawn bounding boxes
  - Reduces total annotation time by 9$$\times$$ to 18$$\times$$.

# Introduction
Typically, object detectors are trained under *full supervision*, which requires manually drawing tight object bounding boxes.
However, manually drawing tight object bounding boxes takes lots of time.
- Full supervision: learning with full object bounding boxes

Object detectors can also be trained under *weak supervision* using only image-level labels.
This is cheaper but the resulting detectors typically deliver only about half the accuracy of their fully supervised counterparts.
- Weak supervision: learning with only image-level labels without object bounding boxes

## Workflow of Collecting Click Annotation

1. Center-click annotation: Annotators click on the center of an imaginary bounding box enclosing the object.
  - Asked two different annotators to get more accurate estimation
  - Given the two clicks, the *size* of the object can be estimated, by exploiting a correlation between the object size and the distance of the click to the true center (error).

![Workflow]({{ site.url }}{{ site.baseurl }}/assets/images/click supervision/workflow.png){: .align-center}
*Figure 1: The workflow of crowd-sourcing framework for collecting click annotations.*
{: .text-center}

2. Incorporate these clicks into a reference Multiple Instance Learning (MIL) framework which was originally designed for weakly supervised object detection.
  - It jointly localizes object bounding boxes over all training images of an object class.
  - It iteratively alternates between retraining the detector and re-localizing objects.
  - We use the center-clicks in the re-localization phase.
  
## Click Supervision
Click annotation schemes have been used
- Part-based detection to annotate part locations of an object
- in human pose estimation
- in reducing the annotation time for semantic segmentation

# Incorporating clicks into WSOL
Incorporating click supervision into a reference Multiple Instance Learning (MIL) framework, which is typically used in WSOL.

## Reference Multiple Instance Learning (MIL)
- Training set contains
  - Positive images which contain the target class
  - Negative images which do not contain the target class
  


# References
- Paper: Training object class detectors with click supervision [[Link](https://arxiv.org/pdf/1704.06189.pdf)]

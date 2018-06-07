---
title: "Paper Skimming"
categories:
  - Research
tags:
  - paper skimming
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

# Scalable Deep Learning Logo Detection
- Conference: Arxiv
- Institute: Queen Mary University of London, Vision Semantics Ltd.

## Summary
- **Problem Statement**
  - Existing logo detection methods usually consider a small number of logo classes and limited images per class with a strong assumption of requiring tedious object bounding box annotations.
  - This is not scalable to real-world dynamic applications.
  
- **Research Objective**
  - To handle the problem by exploring the webly data learning principle without the need for exhaustive manual labelling.
  - To learn scalable logo detection method
  
- **Proposed Solution**
  - Propose a novel incremental learning approach, called Scalable Logo Self-co-Learning (SL<sup>2</sup>)
  - It is capable of automatically self-discovering informative training images from noisy web data for progressively improving model capability in a cross-model co-learning manner.
  
- **Contribution**
  - Introduce a very large (2,190,757 images of 194 logo classes) logo dataset "WebLogo-2M"
  - Proposed SL<sup>2</sup> method is superior over the state-of-the-art and weekly supervised detection and contemporary webly data learning approaches.

![Algorithm]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/logo.png){: .align-center}{:height="80%" width="80%"}
*Figure: Logo detection performance on WebLogo-2M.*
{: .text-center}

## References
- Paper: [Scalable Deep Learning Logo Detection](https://arxiv.org/pdf/1803.11417.pdf)

# What's the point: semantic segmentation with point supervision
- Conference: ICCV2016

## Summary

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

![Segmentation Method]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/segmentation method.png){: .align-center}{:height="100%" width="100%"}
*Figure:(Top): Overview of our semantic segmentation training framework. (Bottom): Different levels of training supervision*
{: .text-center}

## References
- Paper: [What's the point: semantic segmentation with point supervision](http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/bearman16cvpr.pdf)

# Learning a rotation invariant detector with rotatable bounding box
- Conference: CVPR2018

## Summary

- **Problem Statement**
  - Detection of arbitrarily rotated objects is a challenging task due to the difficulties of locating the multi-angle objects and separating them effectively from the background.
  - The existing methods are not robust to angle varies of the objects because of the use of traditional bounding box, which is a rotation variant structure for locating rotated objects.
  
- **Research Objective**
  - To suggest object detector which is rotation invariant
  
- **Proposed Solution**
  - Propose rotatable bounding box (RBox) and detector (DRBox) which can handle the situation where the orientation angles of the objects are arbitrary.
  - The training of DRBox forces the detection networks to learn the correct orientation angle of the objects, so that the rotation invariant property can be achieved.

![RBox]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/rbox.png){: .align-center}{:height="80%" width="80%"}
*Figure: Comparision of traditional bounding box and rotatable bounding box.*
{: .text-center}

![DRBox]({{ site.url }}{{ site.baseurl }}/assets/images/paper skim/drbox.png){: .align-center}{:height="80%" width="80%"}
*Figure: The networks structure of DRBox. The networks structure of DRBox is similar with other box based methods except for the use of multi-angle prior RBoxes. DRBox searches for objects using sliding and rotation prior RBoxes on input image and then output locations of objects besides with their orientation angles.*
{: .text-center}

  
- **Contribution**
  - Compared with *Faster R-CNN* and *CNN*, DRBox performs much better than traditional bounding box based methods do on the given tasks, and is more robust against rotation of input image and target objects.
  - DRBox correctly outputs the orientation angles of the objects.

## References
- [Paper] [Learning a rotation invariant detector with rotatable bounding box](https://arxiv.org/pdf/1711.09405.pdf)
- [Github] [DRBox](https://github.com/liulei01/DRBox)

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
---
title: "BING: Binarized Normed Gradients for Objectness Estimation at 300fps"
categories:
  - Research
tags:
  - Objectness
  - BING
header:
  teaser: /assets/images/bing/qualitative.png
  overlay_image: /assets/images/bing/qualitative.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"BING: Binarized Normed Gradients for Objectness Estimation at 300fps" is a an objectness classifier using binarized normed gradient and linear classifier, which is supported by OpenCV library.

{% include toc title="Table of Contents" icon="file-text" %}

# Summary

- **Goal**
  - Design a good generic objectness measure method 

- **Proposed Solution**
  - Simple idea: collect a few hundred object bounding boxes and a few hundred bounding boxes with random background, and train a linear SVM classifier to predict object vs no-object.
  - We observe that generic objects with well-defined closed boundaries share surprisingly strong correlation when looking at the norm of the gradient, after resizing of their corresponding image windows to small fixed size (e.g. 8x8).
  - In order to efficiently quantify the objectness of an image window, we resize it to 8x8 and use the norm of the gradients as a simple 64D feature for learning a generic objectness measure in a cascaded SVM framework.  

- **Contribution**
  - Propose a simple and powerful feature "BING" to help the search for objects using objectness scores.  
  - Further show how the binarized version of the NG feature, namely binarized normed gradients (BING) feature, can be used for efficient objectness estimation of image windows, which reauires only a few atomic CPU operations.
  - With the PASCAL VOC2007 dataset, BING efficiently (300fps on a single laptop CPU) generates a small set of data-driven, category-independent, high quality object windows, yielding 96.2% detection rate (DR) with 1,000 windows.
  - For the generalization ability, when training BING on 6 object categories and testing on other 14 unseen categories, we observed similar high performance as in standard settings.

# BING

- **Objectness**: a vlaue which reflects how likely an image window covers an object of any category.
- Designing a good generic objectness measure method should:
  - achieve **high object detection rate** (DR), as any undetected objects at this stage cannot be recovered later
  - produce **a small number of proposals** for reducing computational time of subsequent detectors
  - obtain **high computational efficiency** so that the method can be easily involved in various applications, especially for realtime and large-scale applications
  - have **good generalization ability** to unseen object categories, so that the proposals can be reused by many category specific detectors to greatly reduct the computation for each of them

## Observation

![Obs]({{ site.url }}{{ site.baseurl }}/assets/images/bing/NG.png){: .align-center}{:height="100%" width="100%"}
{: .text-center}

  - When resizing windows corresponding to real world objects to a small fixed size, the norm (i.e. magnitude) of the corresponding image gradients gradients becomes a good discriminative feature, because of the little variation that closed boundaries could present in such abstracted view.
  - [Image gradient](https://en.wikipedia.org/wiki/Image_gradient) is a directional change in the intensity or color in an image.

## Learning objectness measurement
1. Firstly, resize the input image to different quantized sizes and calculate the normed gradients of each resized image.
  - The values in an 8x8 region of these resized normed gradients maps are defined as a **64D normed gradients (NG)** feature of its corresponding window.
2. Learn a single model using linear SVM.
  - NG features of the ground truth object windows and random sampled background windows are used as positive and negative training samples respectively.
3. Learn coefficient and bias terms for each quantised size for objectness score

# Experiments

## Computational Time

![Ex1]({{ site.url }}{{ site.baseurl }}/assets/images/bing/computational_time.png){: .align-center}
{: .full}

## Qualitative Analysis

![Ex2]({{ site.url }}{{ site.baseurl }}/assets/images/bing/qualitative.png){: .align-center}{:height="100%" width="100%"}
{: .text-center}

## Quantitative Analysis

![Ex3]({{ site.url }}{{ site.baseurl }}/assets/images/bing/quantitative.png){: .align-center}{:height="100%" width="100%"}
{: .text-center}

## References
- Paper: [BING: Binarized Normed Gradients for Objectness Estimation at 300fps](https://mmcheng.net/mftp/Papers/ObjectnessBING.pdf)
- Github: [Objectness Proposal Generator with BING](https://github.com/torrvision/Objectness)
- [OpenCV Saliency Detection](https://www.pyimagesearch.com/2018/07/16/opencv-saliency-detection/)
- [Objectness Algorithms](https://docs.opencv.org/3.0-beta/modules/saliency/doc/objectness_algorithms.html#objectnessbing-settrainingpath)
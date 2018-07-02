---
title: "Scale-Invariant Feature Transform (SIFT)"
categories:
  - Research
tags:
  - sift
header:
  teaser: /assets/images/sift/dog.png
  overlay_image: /assets/images/sift/dog.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

**Scale-Invariant Feature Transform (SIFT)** is an old algorithm presented in 2004, D.Lowe, University of British Columbia. However, it is one of the most famous algorithm when it comes to distinctive image features and scale-invariant keypoints.

{% include toc title="Table of Contents" icon="file-text" %}

# Summary

- **Problem Statement**
  - Proposed methods before SIFT (e.g. Harris corner) are not invariant to image scale and rotation.
  
- **Research Objective**
  - To find a method for extracting distinctive invariant features from images that can be used to perform reliable matching between different views of an object or scene.
  
- **Proposed Solution**
  1. Scale-space extrema detection
  2. Keypoint localization
  3. Orientation assignment
  4. Keypoint descriptor

- **Contribution**
  - The features are invariant to image scale and rotation, and are shown to provide robust matching across a substantial range of affine distortion, change in 3D viewpoint, addition of noise, and change in illumination.
  - The features are highly distinctive, in the sense that a single feature can be correctly matched with high probability against a large database of features from many images.
  - The authors described an approach to using these features for *object recognition*.
    - It could robustly identify objects among clutter and occlusion while achieving near real-time performance.

# Proposed Method
There are four major stages of computation to generate the set of image reatures.

## 1. Scale-space extrema detection
 First stage searches over all scales and image locations. It is implemented efficiently by using a difference-of-Gaussian function to identify potential interest points that are invariante to scale and orientation.

## 2. Keypoint localization
At each candidate location, a detailed model is fit to determine location and scale. Keypoints are selected based on measures of their stability.

## 3. Orientation assignment
One or more orientations are assigned to each keypoint location based on local image gradient directions. All future operations are performed on image data that has been transformed relative to the assigned orientation, scale, and location for each feature, thereby providing invariance to these transformations.

## 4. Keypoint descriptor
The local image gradients are measured at the selected scale in the region around each keypoint. These are transformed into a representation that allows for significant levels of local shape distortion and change in illumination.

# References
- Paper: [Distinctive image features from scale-invariant keypoints](http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), International Journal of COmputer Vision 2004
  - It gives the most complete and up-to-date reference for the SIFT feature detector
- Paper: [Object recognition from local scale-invariant features](http://www.cs.ubc.ca/~lowe/papers/iccv99.pdf), ICCV 1999
  - It gives the SIFT approach to invariant keypoint detection and some more information on the applications to object recognition
- Paper: [Local feature view clustering for 3D object recognition](http://www.cs.ubc.ca/~lowe/papers/cvpr01.pdf)
  - It gives methods for performing 3D object recogniiton by interpolating between 2D views and a probabilistic model for verification of recognition
- [Wikipedia](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)
- OpenCV: [Introduction to SIFT (Scale-Invariant Feature Transform)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html)
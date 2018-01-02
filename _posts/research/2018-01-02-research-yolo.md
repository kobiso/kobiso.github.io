---
title: "You Only Look Once (YOLO)"
categories:
  - Research
tags:
  - YOLO
  - object detection
header:
  teaser: /assets/images/yolo/yolo9000 result.png
  overlay_image: /assets/images/yolo/yolo9000 result.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

*'You Only Look Once: Unified, Real-Time Object Detection'* (YOLO) proposed an object detection model which was presented at IEEE Conference on Computer Vision and Pattern Recognition in 2016.
And *'YOLO9000: Better, Faster, Stronger'* proposed an improved version of YOLO which was presented at IEEE Conference on Computer Vision and Pattern Recognition in 2017.

{% include toc title="Table of Contents" icon="file-text" %}

# YOLO

## Summary
- **Problem Statement**
  - Prior work on object detection repurposes classifiers to perform detection.
  
- **Research Objective**
  - To boost up the speed of prior object detection method while having similar performance with state-of-the-art detection systems

- **Solution Proposed: YOLO**
  - Frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities
  - A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. 
  
- **Contribution**
  - YOLO is fast since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
  - YOLO reasons gloablly about the image when making predictions
  - YOLO learns very general representations of objects.
  
# YOLO9000

## Summary

# References
- Paper: You Only Look Once: Unified, Real-Time Object Detection [[Link](https://arxiv.org/abs/1506.02640)]
- Paper: YOLO9000: Better, Faster, Stronger [[Link](https://arxiv.org/pdf/1612.08242.pdf)]
---
title: "Rotation Invariance & Equivariance"
categories:
  - Research
tags:
  - paper skimming
  - rotation invariance
  - rotation equivariance
header:
  teaser: /assets/images/rotation/arf1.png
  overlay_image: /assets/images/rotation/arf1.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post is a summary or paper skimming on rotation invariance and equivariance related research.
So, this post will be keep updating by the time.

{% include toc title="Table of Contents" icon="file-text" %}

# Paper List

## Rotated Object Detector
- [Arbitrary-Oriented Scene Text Detection via Rotation Proposals]({{ site.url }}{{ site.baseurl }}/research/research-rotation/#arbitrary-oriented-scene-text-detection-via-rotation-proposals), Transaction on Multimedia 2018
  - [Paper](https://arxiv.org/pdf/1703.01086.pdf)
- [Learning a Rotation Invariant Detector with Rotatable Bounding Box]({{ site.url }}{{ site.baseurl }}/research/research-rotation/#learning-a-rotation-invariant-detector-with-rotatable-bounding-box), CVPR2018 submitted?
  - [Paper](https://arxiv.org/pdf/1711.09405.pdf), [Code-Caffe](https://github.com/liulei01/DRBox), [Related GitHub](https://github.com/uoip/SSD-variants)
- Rotational Rectification Network: Enabling Pedestrian Detection for Mobile Vision, WACV2018.
  - [Paper](http://xinshuoweng.com/papers/R2N/WACV2018_proceedings.pdf), [Oral](http://xinshuoweng.com/papers/R2N/oral.pptx) 

## Feature Representation & Network
- [Oriented Response Networks]({{ site.url }}{{ site.baseurl }}/research/research-rotation/#oriented-response-networks), CVPR2017
  - [Paper](https://arxiv.org/pdf/1701.01833.pdf), [Torch-Code](http://yzhou.work/ORN/)
- [Unsupervised Representation Learning by Predicting Image Rotations]({{ site.url }}{{ site.baseurl }}/research/research-rotation/#unsupervised-representation-learning-by-predicting-image-rotations), ICLR2018
  - [Paper](https://arxiv.org/pdf/1803.07728.pdf)
- Deep Rotation Equivariant Network, CoRR2017
  - [Paper](https://arxiv.org/pdf/1705.08623.pdf), [TF-Code](https://github.com/microljy/DREN)
- Generic 3D Representation via Pose Estimation and Matching, ECCV2016
  - [Paper](http://3drepresentation.stanford.edu/)
- Harmonic Networks: Deep Translation and Rotation Equivariance, CVPR2017
  - [Paper](http://visual.cs.ucl.ac.uk/pubs/harmonicNets/pdfs/worrallEtAl2017.pdf), [Project page](http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html), [Code](https://github.com/deworrall92/harmonicConvolutions)

## Rotation Detection & Correction
- [Blog][Correcting Image Orientation Using Convolutional Neural Networks]({{ site.url }}{{ site.baseurl }}/research/research-rotation/#blog-correcting-image-orientation-using-convolutional-neural-networks)
  - [Blog](https://d4nst.github.io/2017/01/12/image-orientation/), [Code](https://github.com/d4nst/RotNet)

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

![RBox]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rbox.png){: .align-center}{:height="80%" width="80%"}
*Figure: Comparision of traditional bounding box and rotatable bounding box.*
{: .text-center}

![DRBox]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/drbox.png){: .align-center}{:height="80%" width="80%"}
*Figure: The networks structure of DRBox. The networks structure of DRBox is similar with other box based methods except for the use of multi-angle prior RBoxes. DRBox searches for objects using sliding and rotation prior RBoxes on input image and then output locations of objects besides with their orientation angles.*
{: .text-center}

  
- **Contribution**
  - Compared with *Faster R-CNN* and *CNN*, DRBox performs much better than traditional bounding box based methods do on the given tasks, and is more robust against rotation of input image and target objects.
  - DRBox correctly outputs the orientation angles of the objects.

## References
- Paper: [Learning a rotation invariant detector with rotatable bounding box](https://arxiv.org/pdf/1711.09405.pdf)
- Github: [DRBox](https://github.com/liulei01/DRBox)

# Oriented Response Networks
- Conference: CVPR2017

## Summary

- **Problem Statement**
  - Deep Convolution Neural Networks (DCNNs) ability in handling significant local and global image rotations remains limited.
  
- **Research Objective**
  - To suggest orientation invariant deep feature for DCNNs
  
- **Proposed Solution**
  - Propose Active Rotating Filters (ARFs) that actively rotate during convolution and produce feature maps with location and orientation explicitly encoded.
  - An ARF acts as a virtual filter bank containing the filter itself and its multiple unmaterialised rotated versions.

![ARF]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/arf1.png){: .align-center}{:height="80%" width="80%"}
*Figure: An ARF is a filter of the size W X W X N, and viewed as N-directional points on a W X W grid. The form of the ARF enables it to effectively define relative rotations, e.g., the head rotation of a bird about its body. An ARF actively rotates during convolution; thus it acts as a virtual filter bank containing the canonical filter itself and its multiple unmaterialised rotated versions. In this example, the location and orientation of birds in different postures are captured by the ARF and explicitly encoded into a feature map.*
{: .text-center}
  
- **Contribution**
  - Specified Active Rotating Filters and Oriented Response Convolution, improved the most fundamental module of DCNN and endowed DCNN the capability of explicitly encoding hierarchical orientation information.
  - Applied such orientation information to rotation-invariant image classification and object orientation estimation.
  - Upgraded successful DCNNs including VGG, ResNet, TI-Pooling and STN to ORNs, achieving state-of-the-art performance with significantly fewer network parameters on popular benchmarks.

![ARF result]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/arf2.png){: .align-center}{:height="80%" width="80%"}
*Figure: Results on the natural image classification benchmark. In the second column, k is the widening factor corresponding to the number of filters in each layer.*
{: .text-center}


## References
- Paper: [Oriented Response Networks](https://arxiv.org/pdf/1701.01833.pdf)
- [Project page](http://yzhou.work/ORN/)

# Unsupervised representation learning by predicting image rotations
- Conference: ICLR2018

## Summary

- **Problem Statement**
  - In computer vision task, they usually require massive amounts of manually labeled data, which is both expensive and impractical to scale.
  - Therefore, unsupervised semantic feature learning is important.
  
- **Research Objective**
  - To learn ConvNet based semantic features in an unsupervised manner.
  
- **Proposed Solution**
  - Propose to learn image features by training ConvNets to recognize the 2d rotation that is applied to the image that it gets as input.

![url1]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/url1.png){: .align-center}
{: .full}

*Figure: Proposed self-supervised task. Given four possible geometric transformations, the 0, 90, 180, 270 degrees rotations, proposed method train a ConvNet model $$F(.)$$ to recognize the rotation that is applied to the image that it gets as input. $$F^y(X^{y^*})$$ is the probability of rotation transformation $$y$$ predicted by model $$F(.)$$ when it gets as input an image that has been transformed by the rotation transformation $$y^*$$.*
{: .text-center}
  
- **Contribution**
  - Offers a powerful supervisory signal for semantic feature learning
  - In all of various evaluation, proposed self-supervised formulation demonstrates state-of-the-art results with dramatic improvements w.r.t. prior unsupervised approaches.
  - Proposed self-supervised learning approach significantly narrows the gap between unsupervised and supervised feature learning.

![url2]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/url2.png){: .align-center}{:height="100%" width="100%"}
*Figure: Task & Dataset Generalization: PASCAL VOC 2007 classification and detection results, and PASCAL VOC 2012 segmentation results.*
{: .text-center}

## References
- Paper: [Unsupervised representation learning by predicting image rotations](https://arxiv.org/pdf/1803.07728.pdf)
- [Code](https://github.com/gidariss/FeatureLearningRotNet)

# Arbitrary-oriented scene text detection via rotation proposals
- Conference: IEEE Transactions on Multimedia 2018

## Summary

- **Problem Statement**
  - In real-world applications, a large number of the text regions are not horizontal.
  - Thus, recent years' horizontal-specific methods cannot be widely applied in practice.
  
- **Research Objective**
  - To detect arbitrary-oriented text in natural scene images
  
- **Proposed Solution**
  - Propose the *Rotation Region Proposal Networks (RRPN)*, which are designed to generate inclined proposals with text orientation angle information.
  - The angle information is then adapted for bounding box regression to make the proposals more accurately fit into the text region in terms of the orientation.
  - The *Rotation Region-of-Interest (RRoI)* pooling layer is proposed to project arbitrary-oriented proposals to a feature map for a text region classifier.

![Comparison]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotated text detection1.png){: .align-center}{:height="80%" width="80%"}
*Figure: First row: text detection based on horizontal bounding box proposal and bounding box regression of Faster-RCNN. Second row: detection using rotation region proposal and bounding box regression with orientation step.*
{: .text-center}

![Pipeline]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotated text detection2.png){: .align-center}
{: .full}

*Figure: Rotation-based text detection pipeline.*
{: .text-center}
  
- **Contribution**
  - Proposed solution has the ability to predict the orientation of a text line using a region-proposal-based approach; thus, the proposals can better fit the text region, and the ranged text region can be easily rectified and is more convenient for text reading.
  - RRoI pooling layer and learning of the rotated proposal are incorporated into the region-proposal-based architecture, which ensures the computational efficiency of text detection compared with segmentation-based text detection systems.
  - Propose strategies for the refinement of region proposals with arbitrary orientation to improve the performance of arbitrary-oriented text detection

![Result]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotated text detection3.png){: .align-center}
{: .full}

*Figure: Comparison with state-of-the-art approaches on three benchmarks. Bold text denotes the top result, while underlined text corresponds to the second runner-up.*
{: .text-center}


## References
- Paper: [Arbitrary-oriented scene text detection via rotation proposals](https://arxiv.org/pdf/1703.01086.pdf)
- [Code](https://github.com/mjq11302010044/RRPN)

# [Blog] Correcting Image Orientation Using Convolutional Neural Networks

- **Objective**
  - To predict the rotation angle of an image and correct it into upright image

- **Network**
  - Used CNN model
    - Simple CNN for MNIST
    - ResNet50 with pre-trained on ImageNet for Google Street View 
  - The output: we need the network to predict the image's rotation angle, which can then be used to rotate the image in the opposite direction to correct its orientation.
  - Used classification: the network should produce a vector of 360 values instead of a single value. Each value of that vector represents the probability between 0 and 1 of each class being the correct one.
    - The author mentioned that using classification showed better performance than regression.
  - Used *angle_error* as a metric: to monitor the accuracy of the model during training. It will be in charge of periodically computing the angle difference between predicted angles and true angles.
  - Used Keras based on TensorFlow

- **Data set**
  - MNIST and Google Street View

![Comparison]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotnet1.png){: .align-center}{:height="80%" width="80%"}

- **RotNet on MNIST**
  - After 50 epochs, the network achieves an average angle error of 6-7 degrees in the validation set.

![Comparison]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotnet2.png){: .align-center}{:height="100%" width="100%"}

- **RotNet on Google Street View**
  - After 10 epochs to get an average angle error of 1-2 degrees.

![Comparison]({{ site.url }}{{ site.baseurl }}/assets/images/rotation/rotnet3.png){: .align-center}{:height="100%" width="100%"}

## References
- Blog: [Correcting Image Orientation Using Convolutional Neural Networks](https://d4nst.github.io/2017/01/12/image-orientation/)
- Github: [RotNet](https://github.com/d4nst/RotNet)
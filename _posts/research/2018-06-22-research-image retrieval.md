---
title: "Image Retrieval"
categories:
  - Research
tags:
  - paper skimming
  - image retrieval
header:
  teaser: /assets/images/image retrieval/delf1.png
  overlay_image: /assets/images/image retrieval/delf1.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post is a summary and paper skimming on image retrieval related research.
So, this post will be keep updating by the time.

{% include toc title="Table of Contents" icon="file-text" %}

# Paper List

## Feature Representation
- [Large-Scale Image Retrieval with Attentive Deep Local Features]({{ site.url }}{{ site.baseurl }}/research/research-image-retrieval/#large-scale-image-retrieval-with-attentive-deep-local-features), ICCV2017
  - [Paper](https://arxiv.org/pdf/1612.06321.pdf), [Code](https://github.com/tensorflow/models/tree/master/research/delf)

- [Efficient diffusion on region manifolds: recovering small objects with compact CNN representations]({{ site.url }}{{ site.baseurl }}/research/research-image-retrieval/#efficient-diffusion-on-region-manifolds), CVPR2017
  - [Paper](https://arxiv.org/pdf/1611.05113.pdf), [Code-Matlab](https://github.com/ahmetius/diffusion-retrieval)

- [Particular object retrieval with integral max-pooling of cnn activations (RMAC)]({{ site.url }}{{ site.baseurl }}/research/research-image-retrieval/#r-mac), ICLR2016
  - [Paper](https://arxiv.org/pdf/1511.05879)

## Metric Learning
- [Improved deep metric learning with multi-class N-pair loss objective]({{ site.url }}{{ site.baseurl }}/research/research-n-pair-loss/), NIPS2016
  - [Paper](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)

- [Deep image retrieval: learning global representations for image search]({{ site.url }}{{ site.baseurl }}/research/research-deepir/), ECCV2016
  - [Paper](https://arxiv.org/pdf/1604.01325)

# Large-Scale Image Retrieval with Attentive Deep Local Features
- Conference: ICCV2017

## Summary

- **Problem Statement**
  - Performance of CNN-based **global descriptors** for image retrieval may be hindered by a wide variety of challenging conditions observed in large-scale datasets, such as clutter, occlusion, and variations in viewpoint and illumination.
    - Global descriptors lack the ability to find patch-level matches between images.
    - As a result, it is difficult to retrieve images based on partial matching in the presence of occlusion and background clutter.
  - CNN-based **local features** are proposed for patch-level matching.
    - However, these techniques are not optimized specifically for image retrieval since they lack the ability to detect semantically meaningful features, and show limited accuracy in practice.
  
- **Research Objective**
  - To develop a large-scale image retrieval system based on a novel CNN-based feature descriptor
  
- **Proposed Solution**
  - Propose an **attentive local feature descriptor suitable for large-scale image retrieval**, referred to as *DELF* (DEep Local Feature)
  - The new feature is based on convolutional neural networks, which are trained only with image-level annotations on a landmark image dataset.
  - To identify semantically useful local features for image retrieval, we also propose an **attention mechanism for keypoint selection**, which shares most network layers with the descriptor.
  - This framework can be used for image retrieval as a drop-in replacement for other keypoint detectors and descriptors, enabling more accurate feature matching and geometric verification.

![Architecture]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/delf1.png){: .align-center}{:height="80%" width="80%"}
*Figure: Overall architecture of DELF. On the left, it illustrate the pipeline for extraction and selection of DELF. On the right, it illustrates large-scale feature-based retrieval pipeline. DELF for database images are indexed offline. The index supports querying by retrieving nearest neighbor (NN) features, which can be used to rank database images based on geometrically verified matches.*
{: .text-center}

- **Contribution**
  - Proposed system produces reliable confidence scores to reject false positives, in particular, it is robust against queries that have no correct match in the database.
  - The evaluation in Google-Landmarks dataset shows that DELF outperforms existing global and local descriptors by substantial margins.
  - The evaluation in existing datasets shows that DELF achieves excellent performance when combined with global descriptors.

![Result1]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/delf2.png){: .align-center}{:height="80%" width="80%"}
*Figure: Performance evaluation on existing datasets in mAP (%). All results of existing methods are based on our reproduction using public source codes. We tested LIFT only on Oxf5k and Par6k due to its slow speed. (* denotes the results from the original papers.)*
{: .text-center}

![Result2]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/delf3.png){: .align-center}
{: .full}

*Figure: Visualization of feature correspondences between images in query and database using DELF+FT+ATT. DELF successfully matches landmarks and objects in challenging environment including partial occlusion, distracting objects, and background clutter.*
{: .full .text-center}

## References
- Paper: [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/pdf/1612.06321.pdf)
- [Code](https://github.com/tensorflow/models/tree/master/research/delf)

# Efficient diffusion on region manifolds
- Title: Efficient diffusion on region manifolds: recovering small objects with compact CNN representations
- Conference: CVPR2017

## Summary

- **Problem Statement**
  - Query expansion is a popular method to improve the quality of image retrieval with both conventional and CNN representations.
  - It has been so far limited to global image similarity.
  
- **Research Objective**
  - Focuses on diffusion, a mechanism that captures the image manifold in the feature space.

  
- **Proposed Solution**
  - The diffusion is carried out on descriptors of overlapping image regions rather than on a global image descriptor.
  - An efficient off-line stage allows optional reduction in the number of stored regions.
  - In the on-line stage, the proposed handling of unseen queries in the indexing stage removes additional computation to adjust the precomputed data.
  - We perform diffusion through a sparse linear system solver, yielding practical query times well below one second.
  
![Diffusion]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/diffusion1.png){: .align-center}{:height="60%" width="60%"}
*Figure:(Top): Diffusion on a synthetic dataset in $$\mathbb{R}^2$$. Dataset points, query points and their k-nearest neighbors are shown in blue, red, and green respectively. Contour lines correspond to ranking scores after diffusion. In this work, points are region descriptors.*
{: .text-center}

- **Contribution**
  - Introduce a *regional diffusion mechanisum*, which handles one or more query vectors at the same cost. This approach significantly improves retrieval of small objects and cluttered scenes.
  - A new approach to unseen queries with no computational overhead is proposed.
  - Experimentally, it gives a significant boost in performance of image retrieval with compact CNN descriptors on standard benchmarks, especially when the query object covers only a small part of the image.

![Quantitative analysis]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/diffusion2.png){: .align-center}{:height="90%" width="90%"}
*Figure:(Top): Performance comparison to the state of the art. Points at 512D are extracted with VGG and at 2048D with
ResNet101 . Regional diffusion with 5 regions uses GMM.*
{: .text-center}

![Qualitative analysis]({{ site.url }}{{ site.baseurl }}/assets/images/image retrieval/diffusion3.png){: .align-center}
{: .full}

*Figure: Query examples from INSTRE, Oxford, and Paris datasets and retrieved images ranked by decreasing order of ranking difference between global and regional diffusion. We measure precision at the position where each image is retrieved and report this under each image for global(red) and regional(blue) diffusion. Average Precision (AP) is reported per query for the two methods.*
{: .full .text-center}

## References
- Paper: [Efficient diffusion on region manifolds: recovering small objects with compact CNN representations](https://arxiv.org/pdf/1611.05113.pdf)
- Code: [Matlab](https://github.com/ahmetius/diffusion-retrieval)

# R-MAC
- Title: Particular object retrieval with integral max-pooling of cnn activations
- Conference: ICLR2016

## Summary

- **R-MAC**
  - It aggregates several image regions into a compact feature vector of fixed length and is thus robust to scale and translation.
  - This representation can deal with high resolution images of different aspect ratios and obtains a competitive accuracy.

## References
- Paper: [Particular object retrieval with integral max-pooling of cnn activations](https://arxiv.org/pdf/1511.05879), ICLR2016
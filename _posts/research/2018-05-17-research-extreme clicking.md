---
title: "Extreme clicking for efficient object annotation"
categories:
  - Research
tags:
  - click supervision
  - bounding box
  - object annotation
header:
  teaser: /assets/images/extreme clicking/method comparison.png
  overlay_image: /assets/images/extreme clicking/method comparison.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

"Extreme clicking for efficient object annotation" proposes a better way to annotate object bounding boxes with four clicks on the object.
It is a further research of "Training object class detectors with click supervision" which proposes efficient way of annotating bounding boxes with one or two click supervision.
This paper was presented in the International Conference on Computer Vision (ICCV) 2017 by Jasper R. R. Uijlings and Vittorio Ferrari (Google AI Perception).

{% include toc title="Table of Contents" icon="file-text" %}

# Summary

- **Problem Statement**
  - Manually annotating object bounding boxes is important but very time consuming.
  - Clicking on imaginary corners of a tight box around the object is difficult as these corners are often outside the actual object and several adjustments are required.
  
- **Research Objective**
  - To minimize human annotation effort while producing high-quality detectors.
  
- **Proposed Solution**
  - **Extreme clicking**: annotator clicks on four physical points on the object (top, bottom, left-, right-most points)
  
- **Contribution**
  - Annotation time is only 7s per box, 5$$\times$$ faster than the traditional way of drawing boxes.
  - The quality of the boxes is as good as the original ground-truth.
  - Detectors trained on our annotations are as accurate as those trained on the original ground-truth.
  - This paper shows how to incorporate them into GrabCut to obtain more accurate segmentations
  - Semantic segmentations models trained on these segmentations outperform those trained on segmentations derived from bounding boxes.

# Introduction
This paper proposes new scheme called *extreme clicking*: annotator clicks on four extreme points of the object.

![Method comparison]({{ site.url }}{{ site.baseurl }}/assets/images/extreme clicking/method comparison.png){: .align-center}{:height="60%" width="60%"}
*Figure 1: Annotating an instance of motorbike: (a) The conventional way of drawing a bounding box. (b) Proposed extreme clicking scheme*
{: .text-center}

- Advantage of extreme clicking
  + Extreme points are not imaginary, but are well-defined physical points on the object
  + No rectangle is involved, neither real nor imaginary
  + Only a single task is performed by the annotator thus avoiding task switching
  + No separate box adjustment step is required
  + No "submit" button is necessary (it terminates after four clicks)

# Object Segmentation from Extreme Clicks
Extreme clicking results not only in high-quality *bounding box annotations*, but also in four accurate *object boundary points*.

- Thinking of segmenting an object instance in image $$I$$ as a *pixel labeling problem*
  + Each pixel $$p \in I$$ should be labeled as either object ($$l_p = 1$$) or background ($$l_p = 0)
  + A labeling $$L$$ of all pixels represents the segmented object
  + Employ a binary pairwise energy function $$E$$ defined over the pixels and their labels
  + $$U$$: unary potential that evaluates how likely a pixel $$p$$ is to take label $$l_p$$ according to the object and background appearance models
  + $$V$$: encourages smoothness by penalizing neighboring pixels taking different labels

$$
E(L) = \sum _p U(l_p) + \sum _{p,q} V(l_p,l_q)
$$

## Initial Object Surface Estimate
For GrabCut to work well, it is important to have a good initial estimate of the object surface to initialize the appearance model and to clamp certain pixels to object.


# References
- Paper: [Extreme clicking for efficient object annotation](https://arxiv.org/pdf/1708.02750.pdf)
- [Supplement](http://calvin.inf.ed.ac.uk/datasets/center-click-annotations/)
- Reference paper: [“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)

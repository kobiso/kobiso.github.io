---
title: "Probability"
categories:
  - Research
tags:
  - Probability
  - Information theory
header:
  teaser: /assets/images/probability/pdf.png
  overlay_image: /assets/images/probability/pdf.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

Learn about probability which are the basics of artificial intelligence and deep learning.
(This article is still writing...)

{% include toc title="Table of Contents" icon="file-text" %}

# Probability
## Two Perspectives of Machine Learning
![Machine Learning]({{ site.url }}{{ site.baseurl }}/assets/images/probability/machine learning2.png){: .align-center}
Machine learning can be explained in two ways,
1. Machine learning is to find a **'function'** which describes data the best (by deciding the function parameter).
2. Machine learning is to find a **'probability density'** which describes data the best (by deciding the probability density function parameter).
  - e.g. If data was described on gaussian distribution, we have to find the best mean and covariance for the data.
  
## Basics of Probability 
### Random Variable
- A *random variable* is a variable that can take on different values randomly and a description of the states that are possible.
  - It must be coupled with a *probability distribution* that specifies how likely each of these states are
  - Random variables can be discrete or continuous.

## Maximum Likelihood Estimation (MLE)
- **Maximum Likelihood Estimation (MLE)** is a way of parameter estimation or random variable with given observation or data.
  - e.g. Imagine if we want to predict $$p$$ by throwing a coin with the probability of $$p$$ of front and $$1-p$$ of back.
   To compute $$p$$ with MLE, we can just divide the number of fronts by the total number of times.
   
- Consider a set of $$n$$ examples $$X = (x_1, x_2, x_3, \ldots, x_n)$$ drawn independently from the true but unknown data generating distribution but unknown data generating distribution $$f(x)$$.
  - Let $$f(x;\theta)$$ be a parametric family of probability distributions over the same space indexed by $$\theta$$
  - The **likelihood** can be defined as,  
$$
\mathcal L (x_1, x_2, \ldots, x_n;\theta) = \mathcal L (X;\theta) = f(X;\theta) = f(x_1, x_2, \ldots, x_n;\theta)
$${: .text-center}
  - **MLE $$\theta_{ML}$$** which is to maximize the likelihood can be defined as,
  
$$
\theta_{ML} = \arg\max_\theta \mathcal L (X;\theta) = \arg\max_\theta f(X;\theta) = \arg\max_\theta \prod_{i=1}^m f(x^{(i)};\theta)
$${: .text-center}      

## Maximum a Posteriori Estimation (MAP)

# References
- Deep Learning book [[Link](http://www.deeplearningbook.org/)]
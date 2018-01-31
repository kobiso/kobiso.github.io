---
title: "Regularization"
categories:
  - Research
tags:
  - regularization
  - generalization
  - L1 regularization
  - L2 regularization
  - dropout
header:
  teaser: /assets/images/regularization/regularization.png
  overlay_image: /assets/images/regularization/regularization.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

**Regularization** is important technique for preventing overfitting problem while training a learning model.

{% include toc title="Table of Contents" icon="file-text" %}

# Regularization
**Regularization** is a way to prevent overfitting and get a model generalizes the data.
Overfitting problem usually caused by large weight value $$W$$, so common way of regularization is to simply adds the bigger penalty as model complexity increases.
Regularization parameter $$\lambda$$ penalizes all the parameters except intercept so that it will decrease the importance given to higher terms and will bring the model towards less complex equation. 

![Regularization]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/regularization.png){: .align-center}{:height="70%" width="70%"} 

# L1/L2 Regularization
## L1 Regularization
**L1 Regularization** also called *Lasso Regression* adds *absolute value of magnitude* of coefficient as penalty term to the loss function.
If $$\lambda$$ is zero, it will be the same with original loss function.
If $$\lambda$$ is very large, it will add too much weight and it will lead to under-fitting.
So, it is important how $$\lambda$$ is chosen.

![L1]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/l1.png){: .align-center}{:height="45%" width="45%"} 

## L2 Regularization (weight decay)
**L2 Regularization** also called *Ridge Regression* is one of the most commonly used regularization technique. 
It adds *squared magnitude* of coefficient as penalty term to the loss function.
If $$\lambda$$ is zero, it will be the same with original loss function.
If $$\lambda$$ is very large, it will add too much weight and it will lead to under-fitting.
So, it is important how $$\lambda$$ is chosen as well.

![L2]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/l2.png){: .align-center}{:height="45%" width="45%"}

Simply thinking that we add $$\frac{1}{2}\lambda W^2$$ on the loss function and after computing gradient descent, $$W$$ will be updated by,

$$
W \leftarrow \eta(\frac{\partial L}{\partial W} + \lambda W)
$${: .text-center} 

So, $$\lambda W$$ will work as penalty according to the amount of $$W$$.  

# Dropout

# References
- Blog: L1 and L2 Regularization Methods [[Link](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)]
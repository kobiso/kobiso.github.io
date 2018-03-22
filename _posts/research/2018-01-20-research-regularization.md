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
*Figure 1: Example of cost function with regularization term.*
{: .text-center}

# L1/L2 Regularization
## L1 Regularization
**L1 Regularization** also called *Lasso Regression* adds *absolute value of magnitude* of coefficient as penalty term to the loss function.
If $$\lambda$$ is zero, it will be the same with original loss function.
If $$\lambda$$ is very large, it will add too much weight and it will lead to under-fitting.
So, it is important how $$\lambda$$ is chosen.

![L1]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/l1.png){: .align-center}{:height="45%" width="45%"} 
*Figure 2: L1 regularization.*
{: .text-center}

## L2 Regularization (weight decay)
**L2 Regularization** also called *Ridge Regression* is one of the most commonly used regularization technique. 
It adds *squared magnitude* of coefficient as penalty term to the loss function.
If $$\lambda$$ is zero, it will be the same with original loss function.
If $$\lambda$$ is very large, it will add too much weight and it will lead to under-fitting.
So, it is important how $$\lambda$$ is chosen as well.

![L2]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/l2.png){: .align-center}{:height="45%" width="45%"}
*Figure 3: L2 regularization.*
{: .text-center}

Simply thinking that we add $$\frac{1}{2}\lambda W^2$$ on the loss function and after computing gradient descent, $$W$$ will be updated by,

$$
W \leftarrow W - \eta(\frac{\partial L}{\partial W} + \lambda W)
$${: .text-center} 

So, $$\lambda W$$ will work as penalty according to the amount of $$W$$.  

# Dropout
**Dropout** [[N. Srivastava et al., 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)] is one of the simplest and the most powerful regularization techniques.
It prevents units from complex co-adapting by randomly dropping units from the network.

![Dropout]({{ site.url }}{{ site.baseurl }}/assets/images/regularization/dropout.png){: .align-center}
{: .full}
*Figure 4: Comparison between network with dropout and without dropout.*
{: .text-center}

- **In the training stage**
  - It randomly drops units using *dropout rate*
  - It has the effect of sampling a large number of different thinned networks.
  - **Dropout rate**: probability that a unit will be retained.
    - Optimal dropout rate is suggested around 0.5 in hidden layers and close to 1 for in input layer.   
    - As it is a hyperparameter, it can be decided empirically. 
    
- **In the inference stage**
  - A single unthinned network is used (no dropout applied)
  - Weights of the unthinned network has to be multiplied by the dropout rate to keep the output the same as training stage.
  - It approximates the effect of averaging the predictions of all thinned networks from the training process. (as *ensemble learning*)
  
- **Drawbacks of dropout**
  - It increases training time because of noisy parameter updates caused when each training step tries to train a different architecture.

# References
- Blog: L1 and L2 Regularization Methods [[Link](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)]
- Paper: Dropout: A Simple Way to Prevent Neural Networks from Overfitting [[Link](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)]
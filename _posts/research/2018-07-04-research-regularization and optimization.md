---
title: "Regularization and Optimization"
categories:
  - Research
tags:
  - paper skimming
  - image retrieval
header:
  teaser: /assets/images/regularization and optimization/penalizing.png
  overlay_image: /assets/images/regularization and optimization/penalizing.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post is a summary and paper skimming on regularization and optimization.
So, this post will be keep updating by the time.

{% include toc title="Table of Contents" icon="file-text" %}

# Paper List

## Regularization
- [Regularizing neural networks by penalizing confident output distributions]({{ site.url }}{{ site.baseurl }}/research/research-regularization-and-optimization/#regularizing-neural-networks-by-penalizing-confident-output-distributions), ICLR2017, Google, Geoffrey Hinton
  - [Paper](https://arxiv.org/pdf/1701.06548.pdf)

## Optimization
- [Gradient acceleration in activation functions]({{ site.url }}{{ site.baseurl }}/research/research-gradient-acceleration/)
  - [Paper](https://arxiv.org/abs/1806.09783.pdf)
  - Sangchul Hahn, Heeyoul Choi (Handong Global University)
- [Cyclical learning rates for training neural networks]({{ site.url }}{{ site.baseurl }}/research/research-regularization-and-optimization/#cyclical-learning-rates)
  - [Paper](https://arxiv.org/abs/1506.01186)
  - Leslie N. Smith (U.S. Naval Research Laboratory)
- [Super-Convergence: very fast training of neural networks using large learning rates]({{ site.url }}{{ site.baseurl }}/research/research-regularization-and-optimization/#super-convergence)
  - [Paper](https://arxiv.org/abs/1708.07120)
  - Leslie N. Smith (U.S. Naval Research Laboratory), Nicholay Topin (university of Maryland)

# Regularizing neural networks by penalizing confident output distributions
- Conference: ICLR2017

## Summary
 
- **Research Objective**
  - To suggest the wide applicable regularizers
  
- **Proposed Solution**
  - Regularizing neural networks by penalizing low entropy output distributions
  - Penalizing low entropy output distributions acts as a strong regularizer in supervised learning.
  - Connect a maximum entropy based confidence penalty to label smoothing through the direction of the KL divergence.
    - When the prior label distribution is uniform, label smoothing is equivalent to adding the KL divergence between the uniform distribution $$u$$ and the network's predicted distribution $$p_\theta$$ to the negative log-likelihood.
    - By reversing the direction of the KL divergence in equation (1), $$D_{KL}(u \parallel p_\theta(y \mid x))$$, it recovers the confidence penalty.

$$
\mathcal{L}(\theta)=-\sum \log p_\theta (y\mid x)-D_{KL}(u \parallel p_\theta(y \mid x)) \cdots (1)
$$

![Comparision]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/penalizing.png){: .align-center}{:height="100%" width="100%"}
*Figure: Distribution of the magnitude of softmax probabilities on the MNIST validation set. A fully-connected, 2-layer, 1024-unit neural network was trained with dropout (left), label smoothing (center), and the confidence penalty (right). Dropout leads to a softmax distribution where probabilities are either 0 or 1. By contrast, both label smoothing and the confidence penalty lead to smoother output distributions, which results in better generalization.*
{: .text-center}

- **Contribution**
  - Both label smoothing and the confidence penalty improve state-of-the-art models across benchmarks without modifying existing hyperparameters

![Result]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/penalizing2.png){: .align-center}{:height="100%" width="100%"}
*Figure: Test error (%) for permutation-invariant MNIST.*
{: .text-center}

## References
- Paper: [Regularizing neural networks by penalizing confident output distributions](https://arxiv.org/pdf/1701.06548.pdf)

# Cyclical Learning Rates
## Problem Statement
- It is known that the learning rate is the most important hyper-parameter to tune for training deep neural networks.
  - It is well known that too small a learning rate will make a training algorithm converge slowly while too large a learning rate will make the training algorithm diverge.
- Hence, one must experiment with a variety of learning rates and schedules, which can be troublesome work.

## Research Objective
- To eliminates the need to experimentally find the best values and schedule for the global learning rates.

## Proposed Solution
- **Cyclical learning rates**: instead of monotonically decreasing the learning rate, this method lets the learning rate cyclically vary between reasonable boundary values.

![LR policy]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/clr1.png){: .align-center}{:height="50%" width="50%"}
*Figure: Triangular learning rate policy. The blue lines represent learning rate values changing between bounds. The input parameter stepsize is the number of iterations in half a cycle.*
{: .text-center}

- **How can one estimate a good value for the cycle length?**
  - The final accuracy results are actually quite robust to cycle length but experiments show that it often is good to set *stepsize* equal to 2 - 10 times the number of iterations in an epoch.

- **How can one estimate reasonable minimum and maximum boundary values?**
  - "LR range test": run your model for several epochs while letting the learning rate increase linearly between low and high LR values.
  - And find reasonable minimum and maximum boundary values with one training run of the network for a few epochs.

![LR range test]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/clr4.png){: .align-center}{:height="70%" width="70%"}
*Figure: AlexNet LR range test; validation classification accuracy as a function of increasing learning rate. base lr = 0.006 and max lr = 0.014.*
{: .text-center}

## Contribution
- Training with cyclical learning rates instead of fixed values achieves improved classification accuracy without a need to tune and often in fewer iterations.
- This paper also dexcribes a simple way to estimate "reasonable bounds" - linearly increasing the learning rate of the network for a few epochs.

![Exp1]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/clr2.png){: .align-center}
{: .full}

![Exp2]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/clr3.png){: .align-center}
{: .full}

## References
- Paper: [Cyclical learning rates for training neural networks](https://arxiv.org/abs/1506.01186)

# Super Convergence

## Research Objective
- This paper describe a phenomenon, which named "super-convergence", where neural networks can be trained an order of magnitude faster than with standard training methods.

## Proposed Solution
- One of the key elements of super-convergence is training with one learning rate cycle and a large maximum learning rate.

### Super-convergence
- In this work, super-convergence use cyclical learning rates (CLR) and the learning rate range test (LR range test).
- The LR range test can be used to determine if super-convergence is possible for an architecture.

![LR test]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg1.png){: .align-center}
{: .full}

*Figure 1: Comparison of learning rate range test results.*
{: .full .text-center}

- **Fig 1-a.** shows typical curve from a LR range test, where the test accuracy has a distinct peak.
  - The learning rate at this peak is the largest value to use as the maximum learning rate bound when using CLR
  - The minimum learning rate can be chosen by dividing the maximum by a factor of 3 or 4.
- **Fig 1-b.** shows that the test accuracy remains consistently high over this unusual range of large learning rates.
  - This unusual behavior is indicative of potential for super-convergence.

- **1 Cycle**: slight modification of cyclical learning rate policy for super-convergence
  - Always use one cycle that is smaller than the total number of iterations/epochs
  - And allow the learning rate to decrease several orders of magnitude less than the initial learning rate for the remaining iterations.

- The general principle of regularization: **The amount of regularization must be balanced for each dataset and architecture**
  - Recognition of this principle permits general use of super-convergence.
  - Reducing other forms of regularization and regularizing with very large learning rates makes training significantly more efficient.

### Estimating Optimal Learning Rates
- This paper derives a simplification of the second order, Hessian-free optimization method to estimate optimal learning rates.
- The large learning rates indicated by these Figures is caused by small values of Hessian approximation and small values of the Hessian implies that SGD is finding flat and wide local minima.

![LR test]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg2.png){: .align-center}
{: .full}

*Figure 2: Estimated learning rate from the simplified Hessian-free optimization while training. The computed optimal learning rates are in the range from 2 to 6.*
{: .full .text-center}


## Experiments
![Exp2]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg3.png){: .align-center}
{: .full}

*Figure 3: Comparisons of super-convergence to typical training outcome with piecewise constant learning rate schedule.*
{: .full .text-center}

- Fig 3-a provides a comparison of super-convergence with a **reduced number of training samples**.
  - When the amount of training data is limited, the gap in performance between the result of standard training and super-convergence increases.
- Fig 3-b illustrates the results for *Resnet-20* and *Resnet-110*.
  - Resnet-20: CLR 90.4% vs. PC-LR: 88.6%, Resnet-110: CLR: 92.1% vs. PC-LR 91.0%
  - The accuracy increase due to super-convergence is greater for the shallower architectures.

![Exp1]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg4.png){: .align-center}
{: .full}

*Figure 4: Comparisons of super-convergence to over a range of batch sizes. These results show that a large batch size is more effective than a small batch size for super-convergence training.*
{: .full .text-center}

- Fig 4 shows experiments on the effects of larger batch size and the generalization gap (the difference between the training and test accuracies).
  - In Fig 4-a, super-convergence training gives improvement in performance with larger batch sizes.
  - In Fig 4-b, it shows that the generalization gap are approximately equivalent for small and large mini-batch sizes.

![Exp3]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg5.png){: .align-center}
{: .full}

*Figure 5: Final accuracy and standard deviation for various datasets and architectures. The total batch size (TBS) for all of the reported runs was 512. PL = learning rate policy or SS = stepsize in epochs, where two steps are in a cycle, WD = weight decay, CM = cyclical momentum. Either SS or PL is provide in the Table and SS implies the cycle learning rate policy.*
{: .full .text-center}

![Exp4]({{ site.url }}{{ site.baseurl }}/assets/images/regularization and optimization/sg6.png){: .align-center}
{: .full}

*Figure 6: Training resnet and inception architectures on the imagenet dataset with the standard learning rate policy (blue curve) versus a 1cycle policy that displays super-convergence. Illustrates that deep neural networks can be trained much faster (20 versus 100 epochs) than by using the standard training methods.*
{: .full .text-center}

- In the Fig 6, experiments with Imagenet show that reducing regularization in the form of weight decay allows the use of larger learning rates and produces much faster convergence and higher final accuracies.
  - The learning rate varying from 0.05 to 1.0, then down to 0.00005 in 20 epochs.
  - In order to use such large learning rates, it was necessary to reduce the value for weight deay.

## Contribution
- **This paper listed their contributions as:**
  - Systematically investigates a new training methodology with improved speed and performance.
  - Demonstrates that large learning rates regularize training and other forms of regularization must be reduced to maintain an optimal balance of regularization.
  - Derives a simplification of the second order, Hessian-free optimization method to estimate optimal learning rates which demonstrates that large learning rates find wide, flat minima.
  - Demonstrates that the effects of super-convergence are increasingly dramatic when less labeled training data is available.

## References
- Paper: [Super-Convergence: very fast training of neural networks using large learning rates](https://arxiv.org/abs/1708.07120)
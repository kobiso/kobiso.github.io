---
title: "Optimization"
categories:
  - Research
tags:
  - optimization
header:
  teaser: /assets/images/optimization/sgd loss surface.gif
  overlay_image: /assets/images/optimization/sgd loss surface.gif
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

It is always important what kind of **optimization algorithm** to use for training a deep learning model.
According to the optimization algorithm we use, the model can produce better and faster results.

{% include toc title="Table of Contents" icon="file-text" %}

# Optimization Algorithm
**Optimization algorithm** minimize or maximize an *error function* (*objective function* or *loss function*) for updating model's internal learnable parameters $$W$$ and $$b$$.

## Types of Optimization
Optimization Algorithm have 2 major categories:

1. **First Order Optimization Algorithms**: These algorithms minimize or maximize a Loss function $$E(x)$$ using *gradient* values with respect to the parameters such as *Gradient descent*.
The first order derivative tells us whether the function is decreasing or increasing at a particular point.
First order Derivative basically give us a line which is Tangential to a point on its Error Surface.

2. **Second Order Optimization Algorithms**: It uses the second order derivative which is called **Hessian** to minimize or maximize the loss function.
The Hessian is a Matrix of *second order partial derivative*.
Since the second derivative is costly to compute, this is not used much.
The second order derivative gives information whether the first derivative is increasing or decreasing which hints at the function's curvature.
Second Order Derivative provide us with a quadratic surface which touches the curvature of the Error Surface.

- **Gradient and Derivative**
  - The **gradient** is a multi-variable generalization of the derivative.
  - While a **derivative** can be defined on functions of a single variable, for functions of several variables, the gradient takes its place.
  - The gradient is a vector-valued function, as opposed to a derivative, which is scalar-valued.
  - A gradient is represented by a **Jacobian matrix** which is simply a matrix consisting of first order partial derivatives (gradients).
  
- **Which Order Optimization to use?**
  1. Now the **First Order Optimization** techniques are easy to compute and less time consuming, converging pretty fast on large data sets.
  2. The **Second Order Optimization** techniques are faster only when the second order derivative is known.
  Otherwise, these methods are always slower and costly to compute in terms of both time and memory.
    - Although, sometimes **Newton's Second Order Optimization** technique can sometimes outperform *first order gradient descent*
    because second order techniques will not get stuck around paths of slow convergence around **saddle points** whereas gradient descent sometimes gets stuck and does not converges.

# Optimization for Deep Learning
## Gradient Descent
**Gradient Descent** is a first order iterative optimization algorithm for finding the minimum of a function.
To find a local minimum of a function using gradient descent, one takes steps proportional to the *negative* of the gradient of the function at the current point.
If instead one takes steps proportional to the *positive* of the gradient, one approaches a local maximum of the function which is known as **gradient ascent**.

Gradient descent is based on the observation that if the multi-variable function (*error or loss function*) $$F(\mathbf{x})$$ is defined and differentiable in a neighborhood of a point $$\mathbf{a}$$,
 then $$F(\mathbf{x})$$ decreases *fastest* if one goes from $$\mathbf{a}$$ in the direction of the negative gradient of $$F$$ at $$\mathbf{a}$$,  $$-\nabla F(\mathbf{a})$$, where $$\gamma$$ is the learning rate.

$$
\mathbf{a}_{n+1} = \mathbf{a}_n-\gamma\nabla F(\mathbf{a}_n)
$${: .text-center}

- **Backpropagation**
  - Backpropagation process is carrying the *error* terms and updates parameters using *gradient descent*
  - Gradient descent uses the gradient of error function $$F$$ respect to the parameters.
  - It updates the parameters in the opposite direction of the gradient of the loss function.
  
![Gradient Descent]({{ site.url }}{{ site.baseurl }}/assets/images/optimization/gradient descent.png)

- **Learning Rate**
  - Learning rate is an amount of decreasing function of each time.
  - Large learning rate could make it bounce back and forth around the minimum.
  - Small learning rate makes it search with a small step and converge very slow. 
  
- **Drawback of Gradient Descent**
  - The traditional batch gradient descent will calculate the gradient of the whole data set.
  - But it will perform only one update, hence it can be very slow and hard to control for datasets which are very large and don't fit in the memory.
  - Also, it computes redundant updates for large data sets.

![Learning Rate]({{ site.url }}{{ site.baseurl }}/assets/images/optimization/learning rate.png){:height="90%" width="90%"}

## Stochastic Gradient Descent
**Stochastic gradient descent (SGD)** is a stochastic approximation of the gradient descent and iterative method for minimizing an objective function.
First SGD was used with each training example (one batch), but these days, we used mini-batch for better performance which also called *mini batch gradient descent*.

- **Motivation**
  - A recurring problem in machine learning is that large training sets are necessary for good generalization, but it cause more computationally expensive.
  - The cost function often decomposes as a sum over training examples of some per-example loss function.
  - The computational cost of below operation is $$O(m)$$ which the time to take a single gradient step becomes prohibitively long as the training set size $$m$$ grows.
  
$$
\triangledown_{\theta}J(\theta)=\frac{1}{m}\sum_{i=1}{m}\triangledown_{\theta}L(x^i,y^i,\theta)
$${: .text-center}

- **Insight**
  - The insight of SGD is that the gradient is an expectation.
  - The expectation may be approximately estimated using a small set of samples.
  - Specifically, on each step of the algorithm, we can sample a **minibatch** of examples drawn uniformly from the training set.
  - The minibatch size $$m'$$ is typically chosen to be a relatively small number of examples and it is usually held fixed as the training set size $$m$$ grows.
  - We may fit a training set with billions of examples using updates computed on only a hundred examples.
  
The estimate of the gradient using minibatch is formed as, 

$$
g = \frac{1}{m'}\sum_{i=1}{m'}\triangledown_{\theta}L(x^i,y^i,\theta) \\
\theta \leftarrow \theta - \epsilon g
$${: .text-center}

where $$\epsilon$$ is the learning rate.

- **Pros**
  - For a fixed model size, the cost per SGD update does not depend on the training set size $$m$$.
  - So, asymptotic cost of training a model with SGD is $$O(1)$$ as a function of $$m$$.
  - It is much faster and performs one update at a time.
  - Due to the frequent updates, parameters updates have high variance and causes the loss function to fluctuate to different intensities.
  - This is actually a good thing because it helps us discover new and possibly better local minima, whereas *standard gradient descent* will only converge to the minimum of the basin.
  - Mini-batch gradient descent is typically the algorithm of choice when training a neural network nowadays.
  
![SGD]({{ site.url }}{{ site.baseurl }}/assets/images/optimization/stochastic.png){:height="90%" width="90%"}
  
- **Cons**
  - The problem of SGD is that due to the frequent updates and fluctuations, it ultimately complicates that convergence to the exact minimum and will keep overshooting.
  - This can be solve by slowly decreasing the learning rate $$\epsilon$$, SGD shows the same convergence pattern as standard gradient descent.

## Challenges of Gradient Descent
1. Choosing a proper learning rate is difficult.
  - Small learning rate would give slow convergence.
  - Large learning rate can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.
2. The same learning rate applies to all parameter updates.
  - If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.
3. Avoiding getting trapped in the numerous sub-optimal *local minima* and *saddle point* is hard while minimizing highly non-convex error functions.
  - The saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

# Optimizing the Gradient Descent

![SGD optimization on loss surface contours]({{ site.url }}{{ site.baseurl }}/assets/images/optimization/sgd loss surface.gif)
![SGD optimization on saddle point]({{ site.url }}{{ site.baseurl }}/assets/images/optimization/sgd saddle point.gif)

# References
- Book: Deep Learning book [[Link](http://www.deeplearningbook.org/)]
- Blog: Types of Optimization Algorithms used in Neural Networks and Ways to Optimize Gradient Descent [[Link](https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f)]
- Wikipedia: Gradient [[Link](https://en.wikipedia.org/wiki/Gradient)]
- Wikipedia: Gradient Descent [[Link](https://en.wikipedia.org/wiki/Gradient_descent)]
- Wikipedia: Stochastic gradient descent [[Link](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)]
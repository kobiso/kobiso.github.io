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

Probability
===========

# Two Perspectives of Machine Learning
![Machine Learning]({{ site.url }}{{ site.baseurl }}/assets/images/probability/machine learning2.png){: .align-center}
Machine learning can be explained in two ways,
1. Machine learning is to find a **'function'** which describes data the best (by deciding the function parameter).
2. Machine learning is to find a **'probability density'** which describes data the best (by deciding the probability density function parameter).
  - e.g. If data was described on gaussian distribution, we have to find the best mean and covariance for the data.

# Random Variable
- A **random variable** is a variable that can take on different values randomly and a description of the states that are possible.
  - It must be coupled with a *probability distribution* that specifies how likely each of these states are.
  - Random variables can be discrete or continuous.
  
# Probability Distributions
- **Probability distribution** is a description of how likely a set of random variables is to take on each of its possible states.

## Discrete Variables and PMF
- **Probability mass function (PMF)** is a probability distribution function over discrete variables.
  - PMF maps from a state of a random variable to the probability of that random variable taking on that state.
  - Notation: $$P(x = a),$$ or $$x \sim P(x)$$
- PMF can act on many variables at the same time which is known as a **joint probability distribution**.
  - $$P(x=a, y=b)$$ denotes the probability that $$x=a$$ and $$y=b$$ simultaneously.
- PMF must satisfy the following properties:
  - The domain of $$P$$ must be the set of all possible states of $$\tt{x}$$.
  - $$\forall{x} \in \tt{x}, 0\leq P(x) \leq 1\$$ .
  - $$\sum_{x\in\tt{x}} P(x) = 1$$ (property as being *normalized*)
  
## Continuous Variables and PDF
- **Probability density function (PDF)** is a probability distribution function over continuous variables.
- PDF must satisfy the following properties:
  - The domain of $$p$$ must be the set of all possible states of $$x$$.
  - $$\forall{x} \in \tt{x}, 0\leq p(x) \leq 1\$$ .
  - $$\int p(x) dx = 1$$ .
- In the univariate example, The probability that $$x$$ lies in the interval $$[a,b]$$ is given by $$\int_{[a,b]}p(x) ds$$. 
  - Uniform distribution on $$[a,b]$$: $$\tt{x} \sim U(a,b)$$

# Marginal Probability
- **Marginal probability distribution** is a probability distribution over a subset of probability distribution over a set of variables (**joint probability distribution**).
- For **discrete random variables** $$\tt{x}$$, $$\tt{y}$$ and we know $$P(\tt{x}, \tt{y})$$, we can find $$P(\tt{x})$$ with the *sum rule*:

$$
\forall x \in {\tt x}, P({\tt x}= x ) = \sum_y P({\tt x}= x , {\tt y}= y ) 
$${: .text-center}

- For **continuous variables**, we need t ouse integration instead of summation:

$$
p(x) = \int p(x,y)dy 
$${: .text-center}
  
# Conditional Probability
- **Conditional probability** is a probability of some event, given that some other event has happened.
- The conditional probability that $${\tt y} = y$$ given $${\tt x} = x$$ as $$P({\tt y} = y \mid {\tt x} = x)$$ when $$P({\tt x} = x) > 0$$

$$
P({\tt y} = y \mid {\tt x} = x) = \frac{P({\tt y} = y, {\tt x} = x)}{P({\tt x} = x)}
$${: .text-center}

## Chain Rule of Conditional Probabilities
- **Chain rule (product rule)**: Any *joint probability distribution* over many random variables may be decomposed into *conditional distributions* over only one variable:

$$
P(x^{(1)}, ... , x^{(n)}) = P(x^{(1)})\prod_{i=2}^n P(x^{(i)} \mid x^{(1)}, ... , x^{(i-1)})\\
P(a,b,c) = P(a \mid b,c)P(b,c)\\
P(b,c) = P(b \mid c)P(c)\\
P(a,b,c) = P(a \mid b,c)P(b \mid c)P(c)
$${: .text-center}

# Independence and Conditional Independence
# Expectation, Variance and Covariance
# Common Probability Distribution
# Useful Properties of Common Functions
# Bayes' Rule
# Technical Details of Continuous Variables
# Information Theory
# Structured Probabilistic Models

# Maximum Likelihood Estimation (MLE)
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
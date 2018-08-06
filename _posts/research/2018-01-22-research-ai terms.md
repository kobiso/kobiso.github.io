---
title: "AI Related Terms"
categories:
  - Research
tags:
  - autoencoder
header:
  teaser: /assets/images/ai terms/local minima.png
  overlay_image: /assets/images/ai terms/local minima.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post will be about artificial intelligence related terms including linear algebra, probability distribution, machine learning and deep learning

{% include toc title="Table of Contents" icon="file-text" %}

# B
## Bootstrapping
- **Bootstrapping** is any test or metric that relies on random sampling with replacement.
  - It allows assigning measures of accuracy to sample estimates.
  - Bootstrapping is the practice of estimating properties of an estimator by measuring those properties when sampling from an approximating distribution.

- **Bootstrapping sample** is a random sample conducted with replacement.
  - Steps:
  1. Randomly select an observation from the original data.
  2. "Write it down"
  3. "Put it back" (i.e. any observation can be selected more than once)
  - Repeat steps 1-3 $$N$$ times: $$N$$ is the number of observations in the original sample.    

- **Reference**
  - Wikipedia: Bootstrapping (statistics) [[Link](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))]
  - Youtube: Bootstrapping Sampling [Link](https://www.youtube.com/watch?v=tTZybQTE0dw)]

## Blob Detection
In computer vision, **blob detection* methods are aimed at detecting regions in a digital image that differ in properties, such as brightness or color, compared to surrounding regions.
Informally, a blob is a region of an image in which some properties are constant or approximately constant; all the points in a blob can be considered in some sense to be similar to each other.
The most common method for blob detection is *convolution*.

- **Reference**
  - Wikipedia: [Blob detection](https://en.wikipedia.org/wiki/Blob_detection)

# C
## Collaborative Filtering
- **Collaborative filtering (CF)** is a technique used by recommender system and it has two sense, a narrow one and a general one.
  1. In the narrow sense, collaborative filtering is a method of making automatic predictions (*filtering*) about the interests of a user by collecting preferences or taste information from many users (*collaborating*).
  2. In the more general sense, collaborative filtering is the process of filtering for information or patterns using techniques involving collaboration among multiple agents, viewpoints, data sources, etc. 

- CF based on users' past behavior have two categories:
  1. **User-based**: measure the similarity between target users and other users
  2. **Item-based**: measure the similarity between the items that target users rates/interacts with and other items
  
- The key idea behind CF is that similar users share the same interest and that similar items are liked by a user.

- **Reference**
  - Wikipedia: Collaborative Filtering [[Link](https://en.wikipedia.org/wiki/Collaborative_filtering)]
  - Blog: Introduction to Recommender System [[Link](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)]

# D
## Discriminative Model
- **Discriminative model** directly estimate class probability (*posterior probability*) $$p(y \mid x)$$ given input
  - Some methods do not have probabilistic interpretation, but fit a function $$f(x)$$, and assign to classes based on it
  - Focus on the decision boundary
  - More powerful with lots of examples
  - Not designed to use unlabeled data
  - Only supervised tasks

![Generative vs Discriminative1]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/generative-discriminative.jpg){: .align-center}

- **Examples**: Gaussians, Naive-Bayesian, Mixtures of multinomial, Mixtures of Gaussians, Hidden Markov Models, Bayesian Networks, Morkov Random Fields

- Check the [**generative model**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#generative-model) for comparison

- **Reference**
  - Wikipedia: Discriminative Model [[Link](https://en.wikipedia.org/wiki/Discriminative_model)]
  - Youtube: Generative vs. discriminative learning [[Link](https://www.youtube.com/watch?v=XtYMRq7f7KA)]
  
## Deterministic Model
- **Deterministic model** is a mathematical model in which outcomes are precisely determined through known relationships among states and events, without room for random variation.
  - A deterministic model always performs the same way for a given set of same initial conditions.
  - It hypothesized an exact relationship between variables which allows one to make predictions and see how one variable affects the other.

- **Example**: (Deterministic) Neural Network, (Deterministic) Regression model
  
- Check the [**probabilistic model**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#probabilistic-model) for comparison

- **Reference**
  - Wikipedia: Mathematical Model [[Link](https://en.wikipedia.org/wiki/Mathematical_model)]
  - Youtube: Deterministic vs Probabilistic Model [[Link](https://www.youtube.com/watch?v=XLPgHer5Cp8)]

# G

## Generative Model
- **Generative model** compute of *posterior probability* $$p(y \mid x)$$ using **bayes rule** to infer distribution over class given input
  - Model the density of inputs $$x$$ from each class $$p(x \mid y)$$
  - Estimate class prior probability $$p(y)$$
  - Probabilistic model of each class
  - Natural use of unlabeled data
  
$$
p(y \mid x) = \frac{p(x \mid y)p(y)}{p(x)}, \quad p(x) = \sum_y p(y)p(x \mid y)
$${: .text-center}

![Generative vs Discriminative2]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/gd.png){: .align-center}
  
- **Examples**: Logistic Regression, Gaussian Process, Regularization Networks, Support Vector Machines, Neural Networks  
  
- Check the [**discriminative model**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#discriminative-model) for comparison
  
- **Reference**
  - Wikipedia: Generative Model [[Link](https://en.wikipedia.org/wiki/Generative_model)]
  - Youtube: Generative vs. discriminative learning [[Link](https://www.youtube.com/watch?v=XtYMRq7f7KA)]

## Graphical Model

# H

## Hamming Distance
In information theory, the **Hamming distance** between two strings of equal length is the number of positions at which the corresponding symbols are different.
In other words, it measures the minimum number of substitutions required to change one string into the other, or the minimum number of errors that could have transformed one string into the other.
In a more general context, the Hamming distance is one of several string metrics for measuring the edit distance between two sequences.

![Hamming distance]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/hamming distance.png){: .align-center}{:height="60%" width="60%"}
*Figure : Example of hamming distance.*
{: .text-center}

- **Reference**
  - Wikipedia: [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)

## Hard Negative Mining
- **Hard positive cases**: anchor and positive samples that are far apart
- **Hard negative cases**: anchor and negative samples that are close together

-**Hard-mining strategies**: Bootstrapping offers a lot of liberties on how the hard examples are chosen. One could for instance pick a limited number of false positives per image or one could fix a threshold and only pick a false positive if its score is superior to a fixed threshold (0.5 for instance).

- **Hard negative class mining**: greadily selects negative classes in a relatively efficient manner, as opposed to negative "instance" mining. It is executed as follows:
  1. Evaluate embedding vectors: choose randomly a large number of output classes C; for each class, randomly pass a few (one or two) examples to extract their embedding vectors.
  2. Select negative classes: select one class randomly from C classes from step 1. Next, greedily add a new class that violates triplet constraint the most w.r.t. the selected classes till we reach N classes. When a tie appears, we randomly pick one of tied classes.
  3. Finalize N-pair: draw two examples from each selected class from step 2.

- **Reference**
  - Paper: [Smart Mining for Deep Metric Learning](https://arxiv.org/pdf/1704.01285.pdf)
  - Paper: [On the use of deep neural networks for the detection of small vehicles in ortho-images](https://hal.archives-ouvertes.fr/hal-01527906/document)
  - Paper: [Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)

# I
## Image Localization, Detection, Segmentation

![Image]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/image.png){: .align-center}

- **Reference**
  - Slide: Stanford CS231 [[Link](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)]

## Invariance vs. Equivariance
**Invariance** to a transformation means if you take the input and transform it then the representation you get the same as the representation of the original, i.e. represent(x) = represent(transform(x))

**Equivariance** to a transformation means if you take the input and transform it then the representation you get is a transformation of the representation of the original, i.e. transform(represent(x)) = represent(transform(x)).

Replicated feature detectors are equivariant with respect to translation, which means that if you translate the input then the representation you get is a translation of the representation of the original.

- **Reference**
  - Paper: [Deep rotation equivariant network](https://arxiv.org/abs/1705.08623)
  - Reddit: [Difference between invariance and equivariance(in terms of convolutional neural networks)](https://www.reddit.com/r/MachineLearning/comments/2q01z5/difference_between_invariance_and_equivariancein/)

# L
## Latent Variable
- **Latent variables** are variables that are not directly observed but are rather inferred from other variables that are observed (directly measured).

- **Reference**
  - Wikipedia: Latent variable [[Link](https://en.wikipedia.org/wiki/Latent_variable)]
  
## Latent Variable Model
- **Latent variable model** is a statistical model that relates a set of observable variables to a set of latent variables.
  - It aims to explain observed variables in terms of latent variables.

- **Reference**
  - Wikipedia: Latent variable model [[Link](https://en.wikipedia.org/wiki/Latent_variable_model)]
  
## Local Minimum Problem
- **Local minimum problem** happens when the backpropagation network converge into a *local minima* instead of the desired **global minimum** since the error value is very complex function with many parameter values of weights.
  - The backpropagation algorithm employs gradient descent by following the slope of error value downward along with the change in all the weight values.
  - The weight values are constantly adjusted until the error value is no longer decreasing.

![Local Minima]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/local minima.png){: .align-center}
  
- Local minimum problem can be avoided with several solutions
  - Utilize **momentum** which gradually increases the weight adjustment rate
  - Use **stochastic gradient descent** is more likely to jump out of a local minimum and find a global minimum, but still can get stuck in local minimum.
    - In stochastic gradient descent, the parameters are estimated for every observation (mini-batch), as opposed the whole sample (full-batch) in regular gradient descent. 
    - This gives lots of randomness and the path of stochastic gradient descent wanders over more places.
  - Adding noise to the weights while being updated
  
- Check the [**saddle point**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#saddle-point) for comparison

- **Reference**
  - ECE Edu: Local Minimum problem [[Link](http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node23.html)]
  
# M

## Metric Learning
**Metric Learning** is the task of learning a distance function over objects.
A metric or distance function has to obey four axioms: non-negativity, Identity of indiscernibles, symmetry and subadditivity / triangle inequality. In practice, metric learning algorithms ignore the condition of identity of indiscernibles and learn a pseudo-metric.

![Metric learning]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/metric learning.png){: .align-center}{:height="80%" width="80%"}
*Figure: Example of metric learning application*
{: .text-center}

- **Reference**
  - Wikipedia: [Similarity learning](https://en.wikipedia.org/wiki/Mean_squared_error)

## Mean Squared Error
**Mean squared error(MSE)** or **mean squared deviation (MSD)** of an estimator measures the average of the squares of the errors or deviations.
The MSE is a measure of the quality of an estimator which is always non-negative, and values closer to zero are better.

- **Predictor**
  - If $$\hat{Y}$$ is a vector of $$n$$ predictions, and $$Y$$ is the vector of observed values of the variable being predicted
  - The MSE is the *mean* $$\left(\frac{1}{n}\sum_{i=1}^n \right)$$ of the *squares of the errors* ($$(Y_i-\hat{Y_i})^2$$)
  
$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^n(Y_i-\hat{Y_i})^2.
$${: .text-center}

- **Estimator**
  - The MSE of an estimator $$\hat{\theta}$$ with respect to an unknown parameter $$\theta$$ is defined as

$$
\operatorname{MSE}(\hat{\theta})=\operatorname{E}_{\hat{\theta}}\left[(\hat{\theta}-\theta)^2\right] \\
\qquad\qquad\qquad\qquad = \operatorname{Var}_{\hat{\theta}}(\hat{\theta})+ \operatorname{Bias}(\hat{\theta},\theta)^2.
$${: .text-cetner}

- **Reference**
  - Wikipedia: Mean squared error [[Link](https://en.wikipedia.org/wiki/Mean_squared_error)]
  
## Multiple Instance Learning (MIL)
**Multiple instance learning (MIL)** is a variation on supervised learning.
Instead of receiving a set of instances which are individually labeled, the learner receives a set of labeled bags, each containing many instances.

Given an image, we want to know its target class based on its visual content.
For instance, the target might be "beach", where the image contains both "sand" and "water"
In MIL terms, the image is described as a bag $$X={X_1, \cdots, X_N}$$, where each $$X_i$$ is the feature vector (called *instance*) extracted from the corresponding $$i$$-th region in the image
 and $$N$$ is the total regions (instances) partitioning the image.
The bag is labeled *positive* ("beach") if it contains both "sand" region instances and "water" region instances.

- **Reference**
  - Wikipedia: Multiple-instance learning [[Link](https://en.wikipedia.org/wiki/Multiple-instance_learning)]

# O

# Object Proposal
**Object proposal** is a hypothesis that is proposed which is not yet a successful detection but just a proposal that needs to be verified and refined.
  - It can be wrong but the trick is to reduce the chances of object detection being wrong.

- **Reference**
  - Quora: What's the difference between object detection and object proposals? [[Link](https://www.quora.com/Whats-the-difference-between-object-detection-and-object-proposals)]

# P
## Principal Component Analysis (PCA)
**Principal component analysis (PCA)** is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.
  - PCA is a mathematical procedure that transforms a number of (possibly) correlated variables into a (smaller) number of uncorrelated variables called **principal components**.
  
![PCA]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/pca.png){: .align-center}
{: .full}

- **Reference**
  - Wikipedia: Principal component analysis [[Link](https://en.wikipedia.org/wiki/Principal_component_analysis)]
  - Blog: Principal component analysis explained visually [[Link](http://setosa.io/ev/principal-component-analysis/)]
  
## Probabilistic Model
- **Probabilistic (stochastic) model** is a mathematical representation of a random phenomenon which is defined by its sample space, events within the sample space, and probabilities.
  - Unlike the deterministic model, the probabilistic model includes elements of randomness.
  - This model is likely to produce different results even with the same initial conditions.
  - There is always an element of chance or uncertainty involved which implies that there are possible alternate solutions.
  
- **Example**: (Probabilistic) Regression models, Probability tress, Monte Carlo, Markov models, Stochastic Neural Network
  
- Check the [**deterministic model**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#deterministic-model) for comparison
  
- **Reference**
  - Wikipedia: Mathematical Model [[Link](https://en.wikipedia.org/wiki/Mathematical_model)]
  - Youtube: Deterministic vs Probabilistic Model [[Link](https://www.youtube.com/watch?v=XLPgHer5Cp8)]
  
# R
## RIP vs. ROP
- **Rotation-In-Plane (RIP)**: Rotation within a plane in two dimensional space

- **Rotation-Off-Plane (ROP)**: Spatial rotation in three dimensional space

![RIP and ROP]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/rip rop.png){: .align-center}{:height="80%" width="80%"}
*Figure: Faces of RIP, ROP, respectively, and concomitance of both RIP and ROP.*
{: .text-center}

- **Reference**
  - Paper: [High-performance rotation invariant multiview face detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4107571)

# S
## Saddle Point
- **Saddle point** is a point on the surface of a function where the slopes (derivatives) of orthogonal function components defining the surface become zero but are not a local extremum on both axes.
  - The critical point with a relative minimum along one axial direction and at a relative maximum along the other axial direction.
  - When we optimize neural networks, for most of the trajectory we optimize, the *critical points* (the points where the derivative is zero or close to zero) are saddle points.
  - Saddle points, unlike local minima, are easily escapable
  
![Saddle Point]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/saddle.png){: .align-center}

- Check the [**local minimum problem**]({{ site.url }}{{ site.baseurl }}/research/research-ai-terms/#local-minimum-problem) for comparison

- **Reference**
  - Wikipedia: Saddle point [[Link](https://en.wikipedia.org/wiki/Saddle_point)]
  
## Sparse Data
- **Sparse data** is data that is easily compressed.
  - Depending on the type of data that you're working with, it usually involves empty slots where data would go.
  - Matrices, for instance, that are have lots of zeroes can be compressed and take up significantly less space in memory.

- **Reference**
  - Web: Data Modeling - What means "Data is dense/sparse" ? [[Link](https://gerardnico.com/wiki/data/modeling/dense_sparse)]
  
# W
## Weakly Supervised Learning
- **Weakly supervised learning** is a term covering a variety of studies that attempt to construct predictive models by learning with weak supervision (incomplete, inexact and inaccurate supervision).
  - It can be supervision with noisy labels. (e.g. bootstrapping, where the bootstrapping procedure may mislabel some examples)
 
- **Three types of weak supervision**
  1. **Incomplete supervision**: where only a subset of training data is given with labels
  2. **Inexact supervision**: where the training data are given with only coarse-grained labels
  3. **Inaccurate supervision**: where the given labels are not always ground-truth  
  
- **Bootstrapping** is one of weakly supervised learning method, which is also called self-training, it is a form of learning that is designed to use less training examples.
  - It starts with a few training examples, trains a classifier, and uses thought-to-be positive examples as yielded by this classifier for retraining.
  - As the set of training examples grow, the classifier improves, provided that not too many negative examples are misclassified as positive, which could lead to deterioration of performance.

![Weakly]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/weakly.png){: .align-center}
{: .full}

![Supervision]({{ site.url }}{{ site.baseurl }}/assets/images/ai terms/weak supervision.png){: .align-center}{:height="90%" width="90%"}
*Figure: Hierarchy tree of supervision.*
{: .text-center}

- **Reference**
  - Stack Overflow: What is weakly supervised learning (bootstrapping)? [[Link](https://stackoverflow.com/questions/18944805/what-is-weakly-supervised-learning-bootstrapping)]
  - Paper: A brief introduction to weakly supervised learning [[Link](https://watermark.silverchair.com/nwx106.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAegwggHkBgkqhkiG9w0BBwagggHVMIIB0QIBADCCAcoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMMLy1On1YQARqVEpQAgEQgIIBm_eusvyUAgChrY5LWDKOAD9z_1qliTEzSJqbFJMufOOE7b7h7L5IG1Qh0sCIS5ZUuwnrQBSDQwpBxQrc2rLL8vMw_Z1iZJCK1rAhglqbD2ZckZ9fcvJ-x1FAYR3ewOpc2udHGv_CUB2A9GHRRD7vR-z4yjBCXPqyBWjn6LVwttPOygFjPKUrxhhrfJKX05o3iNbn0HBQ75skLkLe4BpWHrSDWLnGWMSJFSoLPFmK2xKwBuQJdct9kMeTOFVQxJxDe0VaGTmD_BC0o_YLaH9eXNtC7UqV6yTO0ddURj_Zwwgf09FkpeJNZsaQHnNpOJVNvDOSxA_Go8GkSv-6lgMtBc1OB5zlOAGHqeNPX4BxnIsHIYKS3yJfbx9qdSjG93s-wZyJUqG17eq2JGFEnj4vJ7H7NoPkR8KJRcZrddYfCx__XRYWnd8g8hOnaqpZF14G3nK5zfrf1L6YW-XNRsq0pmiRSyp9VN_cBNM1v4vk1Y_D0_vn7uF3mLkx6nfiJzfljrIfR4R7Ki3TmIdeuRf0KMMgLVGWxGsjAHdp7w)]

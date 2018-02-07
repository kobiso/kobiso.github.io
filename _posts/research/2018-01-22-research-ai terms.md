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

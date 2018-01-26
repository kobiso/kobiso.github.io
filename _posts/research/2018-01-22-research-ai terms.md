---
title: "AI Related Terms"
categories:
  - Research
tags:
  - autoencoder
header:
  teaser: /assets/images/ai terms/pca.png
  overlay_image: /assets/images/ai terms/pca.png
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

This post will be about artificial intelligence related terms including linear algebra, probability distribution, machine learning and deep learning

{% include toc title="Table of Contents" icon="file-text" %}

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
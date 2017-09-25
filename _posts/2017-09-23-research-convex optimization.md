---
title: "Convex optimization problem"
header:
  image: assets/images/unsplash-image-9.jpg
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
categories:
  - Research
tags:
  - convex optimization problem
  - gradient descent method
---

{% include toc title="Table of Contents" icon="file-text" %}

# Convex optimization problem
When we solve machine learning problem, we have to optimize a certain objective function. One of the case of it is convex optimization problem which is a problem of minimizing convex functions over convex sets.

## Optimization problem
> In mathematics and computer science, an **optimization problem** is the problem of finding the best solution from all feasible solutions.

We can write the standard form of a optimization problem as,
$$
\underset{x}{\operatorname{minimize}} \hspace{1em} f(x) \\
\operatorname{subject\;to} \hspace{1em} g_i(x) \leq 0, \quad i = 1,\dots,m \\
\hspace{5em} h_i(x) = 0, \quad i = 1, \dots,p  \\
$$
The function $$f(x)$$ is an objective function to be minimized over the variable $$x$$, and both functions $$g_i(x)$$ and $$h_i(x)$$ are constraints function. If the optimization is maximization problem, it can be treated by negating the objective function. We can think of it as finding an optimum point which can be the minimum or maximum point of the objective function. Sadly, we can not find optimum point in every case. The simplest way to find the optimum point is to find zero point of its derivative function, however, there can be non-differentiable functions or it can not be a extreme point even though it is zero point, such as saddle point. And one of the easy case to find the extreme point is convex optimization.

## Convex function
> In mathematics, a **convex function** is if its epigraph (the set of points on or above the graph of the function) is a convex set.

![Convex function on an interval](https://en.wikipedia.org/wiki/Convex_function#/media/File:ConvexFunction.svg "convex function")
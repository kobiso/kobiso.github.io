---
title: "Convex optimization problem"
categories:
  - Research
tags:
  - convex optimization problem
  - gradient descent method
---

When we solve machine learning problem, we have to optimize a certain objective function. One of the case of it is convex optimization problem which is a problem of minimizing convex functions over convex sets.

{% include toc title="Table of Contents" icon="file-text" %}

## Optimization problem
> In mathematics and computer science, an **optimization problem** is the problem of finding the best solution from all feasible solutions.

We can write the standard form of a optimization problem as,

$$
\underset{x}{\operatorname{minimize}} \hspace{1em} f(x) \\
\operatorname{subject\;to} \hspace{1em} g_i(x) \leq 0, \quad i = 1,\dots,m \\
\hspace{5em} h_i(x) = 0, \quad i = 1, \dots,p  \\
$${: .text-center}


The function $$f(x)$$ is an objective function to be minimized over the variable $$x$$, and both functions $$g_i(x)$$ and $$h_i(x)$$ are constraints function. If the optimization is maximization problem, it can be treated by negating the objective function. We can think of it as finding an optimum point which can be the minimum or maximum point of the objective function. Sadly, we can not find optimum point in every case. The simplest way to find the optimum point is to find zero point of its derivative function, however, there can be non-differentiable functions or it can not be a extreme point even though it is zero point, such as saddle point. And one of the easy case to find the extreme point is convex optimization.

## Convex function
> In mathematics, a **convex function** is if its epigraph (the set of points on or above the graph of the function) is a convex set.

In here, convex set includes a convex region where, for every pair of points within the region, every point on the straight line segment that joins the pair of points is also within the region.The convex function has convex set as a domain of it such as the quadratic function $$x^{2}$$ and the exponential function $$e^{x}$$. The convex function can be written as,

$$
\forall x_1, x_2 \in X, \forall t \in [0, 1]: \qquad f(tx_1+(1-t)x_2)\leq t f(x_1)+(1-t)f(x_2).
$${: .text-center}

![Convex function on an interval]({{ site.url }}{{ site.baseurl }}/assets/images/convex_function.png)

The reason why convex function is important on optimization problem is that it makes optimization easier than the general case since local minimum must be a global minimum. In other word, the convex function has to have only one optimal value, but the optimal point does not have to be one. The below loosely convex function has one optimal value with multiple optimal points.

![Loosely convex function]({{ site.url }}{{ site.baseurl }}/assets/images/loosely_convex_function.png)

If you want to make it one optimal value with only one optimal point, you can put more condition as below.

$$
\forall x_1 \neq x_2 \in X, \forall t \in (0, 1): \qquad f(tx_1+(1-t)x_2) < t f(x_1)+(1-t)f(x_2).
$${: .text-center}

This function is called strictly convex function and we can design an optimization algorithm since it has unique optimal point.

## Notices
Still writing :D
{: .notice}

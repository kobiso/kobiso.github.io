---
title: "Activation functions"
categories:
  - Research
tags:
  - activation function
header:
  teaser: /assets/images/activation functions/sigmoid function.png
  overlay_image: /assets/images/activation functions/sigmoid function.png
  overlay_filter: 0.4
---

Let's talk about activation function in artificial neural network and some questions related of it.

{% include toc title="Table of Contents" icon="file-text" %}

# Activation Function
> **Activation function** of a node defines the output of that node given an input or set of inputs.
It decide whether the node will be "ON" (1) or "OFF" (0) depending on input.
However, only nonlinear activation functions allow such networks to compute nontrivial problems using only a small number of nodes.
In artificial neural networks, this function is also called the **transfer function**.
\- [Wikipedia](https://en.wikipedia.org/wiki/Activation_function) 

$$
Y =  \sum(weight * input) + bias \quad (1)
$${: .text-center}

As it is described in the equation (1), artificial neural network calculates a weighted sum of input and adds a bias.
However, since the range of value $$Y$$ is between $$-\infty$$ and $$+\infty$$, the neuron does not know the bounds of the value and when to "ON" or "OFF" the neuron itself.
This is why we use activation function to make a bound of the value $$Y$$ and decide whether it should be "ON" (1) (activated) or "OFF" (0) (deactivated).
(The idea of neuron to be activated or deactivated is from how our brain and neuron on the brain works.)

# Binary Step Function

$$
f(x) = \begin{cases}
	0 & \text{for  } x < 0\\
	1 & \text{for  } x \ge 0\end{cases}
$${: .text-center}

![Binary step function]({{ site.url }}{{ site.baseurl }}/assets/images/activation functions/step function.svg){: .align-center}

The easiest activation function we can think of is binary step function.
The neuron can be activated (1) if $$Y \ge 0$$, deactivated (0) otherwise.
This kind of activation function will work for binary classification problem which has to decide 1 or 0.
However, if you have to classify more than two classes, this would not a good choice to use.
For example, if you have classify four classes, class1, class2, class3, class4, and it gives you 1 for class1, class2 and 0 for class3, class4, what would you choose for the result?
As the output 1 indicates 100% activated, it is hard to choose between class1 and class2 with valid reason.
Therefore, we rather need to get probabilistic activation output such as, 50% activated and 70% activated.
Alternatively, we need strength or probabilistic activation values rather than just binary activated or deactivated.

# Identity and Linear Function

$$
f(x) = cx \quad (2)
$${: .text-center}

![Linear function]({{ site.url }}{{ site.baseurl }}/assets/images/activation functions/linear function.png){: .align-center}

Linear function or identity function is the first thing we can try to use to handle the problem of binary step function.
Using linear function makes the output from the activation function proportional to input and this gives a range of activations which is not binary.
So, if there are more than one neuron is "ON", we can choose one based on taking the max or softmax.

However, using linear function as activation function has few problems.
First, no matter how many layers we stack on neural network, if all are linear function for each layer, the final activation function of the last layer will be just a linear function of the input of first layer.
It means all the layers we stacked can be replaced by a single layer and it will make stacking layers meaningless.

Second, as the derivative of equation (2) with respect to $$x$$ is $$c$$, this gradient gives no relationship with $$x$$ when we use gradient descent algorithm for training.
Even though we have an error in prediction, the changes made by back propagation will be constant and it does not depend on the change in input $$ \delta(x)$$.
This is a huge problem for training a network since the network need to be trained based on the input and its error in prediction, not constant value.

# Sigmoid Function

$$
f(x) =  \frac{1}{1+e^{-x} } 
$${: .text-center}

![Sigmoid function]({{ site.url }}{{ site.baseurl }}/assets/images/activation functions/sigmoid function.png){: .align-center}

# References
- Activation function in Wikipedia [[Link](https://en.wikipedia.org/wiki/Activation_function)]
- Blog post about activation function[[Link](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)]
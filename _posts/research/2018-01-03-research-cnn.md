---
title: "Convolutional Neural Network"
categories:
  - Research
tags:
  - CNN
header:
  teaser: /assets/images/cnn/activations.jpg
  overlay_image: /assets/images/cnn/activations.jpg
  overlay_filter: 0.4
sidebar:
  nav: "dl"
author_profile: false
---

Learn the basics about Convolutional Neural Network (CNN), its detail and case models of CNN.

{% include toc title="Table of Contents" icon="file-text" %}

# Comparison Between CNN and FFNN
In the figure below, left is a regular 3-layer neural network and right is a CNN arranges its neurons in three dimensions (width, height, depth).
The red input layer in CNN holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels). 

![Network comparison]({{ site.url }}{{ site.baseurl }}/assets/images/cnn/network comparison.png){: .align-center}
{: .full}

- **Regular neural network do not scale well to full images**
  - Fully-connected structure does not scale to larger image. (e.g. $$200 \times 200 \times 3$$ would lead lead to neurons that have 120,000 weights).
  
- **3D volumes of neurons**
  - CNN have neurons arranged in 3 dimensions: width, height, depth (depth: third dimension of an activation volume).
  - The neurons in a layer will only be connected to a small region of the layer, instead of all of the neurons in a fully-connected manner.
  - CNN will reduce the full image into a single vector of class scores, arranged along the depth dimension.
  
# CNN Layers
Convolutional neural network usually use three main types of layers: **Convolutional Layer, Pooling Layer, Fully-Connected Layer**.

- Example architecture for overview: a simple CNN for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]
  - **INPUT** [32x32x3] will holdl the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
  - **CONV** layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to the input volume.
  This may result in volume such as [32x32x12] if we decided to use 12 filters.
  - **RELU** layer will apply an elementwise activation function, such as the $$max(0,x)$$ thresholding at zero. This leaves the size of the volume unchanged [32x32x12].
  - **POOL** layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
  - **FC** layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score as of CIFAR-10.
  
- Notes
  - Each layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don't)
    - These parameters will be traned with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image.
  - Each layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn't)
  - Below figure is an example activations of CNN where the initial volume stores the raw image pixels and the last volume stores the class scores.
  
![Activations of CNN]({{ site.url }}{{ site.baseurl }}/assets/images/cnn/activations.jpg){: .align-center}
{: .full}
  
## Convolutional Layer

- **Local Connectivity**
  - CNN connect each neuron to only a local region of the input volume as It is impractical to connect neurons to all neurons in the previous volume for high-dimensional inputs such as images.
  - The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently to **filter size**).
  - The extent of the connectivity along the depth axis is always equal to the depth of the input volume.
  - An example is shown in below figure where each neuron is connected only to a local region with full depth (i.e. all color channels) and there are multiple neurons (5 in this example) all looking the same region in the input.
![Local connectivity]({{ site.url }}{{ site.baseurl }}/assets/images/cnn/local connectivity.jpg){: .align-center}
  - As shown in below figure, the neurons still compute a dot product of their weights with the input followed by a non-linearity (same with FFNN), but their connectivity is now restricted to be local spatially.
![Local connectivity2]({{ site.url }}{{ site.baseurl }}/assets/images/cnn/local connectivity2.jpg){: .align-center}

- **Spatial Arrangement**
  - Three hyperparameters control the size of the output volume: **the depth**, **stride** and **zero-padding**.
  - **Depth**: It corresponds to the **number of filters**, each learning to look for something different in the input such as, presence of various oriented edges, or blobs of color.
    - Refer to a set of neurons that are all looking at the same region of the input as a depth column
  - **Stride**: it is for sliding the filter, when the stride is 2 then the filters jump 2 pixels at a time.
  - **Zero-padding**: Sometimes it will be convenient to pad the input volume with zeros around the border.
    - It allows us to control the spatial size of the output volumes.
  - Formula to compute the spatial size of the output volume: $$(W-F+2P)/S+1$$
    - where input volume size (W), the receptive field size (F), the stride (S), the amount of zero padding (P)
    
![Spatial arrangement]({{ site.url }}{{ site.baseurl }}/assets/images/cnn/spatial arrangement.png){: .align-center}
 {: .full}
    

## Pooling Layer
## Fully-Connected Layer  

# References
- Standfard CS231n lecture note [[Link](http://cs231n.github.io/convolutional-networks/#pool)]
- Deep Learning book [[Link](http://www.deeplearningbook.org/)]
---
layout: post
title: Overview of Optimizer and Scheduler Architecture in Neural Nets
subtitle: A Gentle Guide, in Plain English
imagecaption: Photo by [](https://unsplash.com/@hirmin) on [Unsplash](https://unsplash.com) 
categories: [ Neural, tutorial ]
author: ketan
tags: featured
image: https://images.unsplash.com/photo-1571156425562-12341e7c9aae?w=1200
---

You prepare the data, create your model architecture and choose the loss function, and are now ready to train it. To train the model, you have to specify an Optimizer and, optionally, a Scheduler.

This is a single line of code. So a lot of the time, you simply choose your "favorite" optimizer that you use for most of your projects - probably SGD or Adam. And forget about it. For many simpler applications that works just fine.

But what does an Optimizer do? And where does it fit into a neural network architecture?

More importantly, what 'knobs' does it have to control its behavior and to tune it to improve the performance of your model?

An Optimizer is defined by three arguments:
- An optimization algorithm eg. SGD, RMSProp, Momentum, ...
- Optimization hyperparameters
- Optimization training parameters

I have an article about #1. Today we will explore #2 and #3.

Optimizers are a critical component of a Neural Network architecture. During training, they play a key role in helping the network learn to make better and better predictions.

In order to explain these topics, we first need to understand a few fundamental concepts. We'll start with a quick introduction to the role that Optimizers play in a deep learning architecture. This is probably stuff you already know, but please be patient, we'll need these so we can build upon them when we get to the more interesting sections :-)

## Quick Primer on Optimization

At a very high level, a neural network executes these steps over numerous iterations during training:
- A Forward pass to generate outputs based on the current parameters and the input data
- A Loss function to calculate a 'cost' for the gap between the current outputs and the desired target outputs
- A Backward pass to calculate the gradients of the loss relative to the parameters
- An Optimization step that uses the gradients to update the parameters so as to reduce the loss for the next iteration

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Network-1.png)
*Components of a network and steps during training (Image by Author)*

What exactly are these parameters since they play such an important role? 

## Model Parameters
A network architecture is built of layers, each of which has some parameters. For example, a Linear or Convolutional layer has a weight and bias parameter. You can also create your own custom layers and define its parameters. A model parameter is a tensor. Like all tensors they contain a matrix of numbers, but they have a special behavior. They have associated gradients which are automatically computed by the framework whenever an operation is performed on a parameter in the forward pass.

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Param-1.png)
*A model consisting of layers, that contain parameters. Another box called 'Model Parameters' that contains 'pointers' to all the parameters (Image by Author)*

Deep Learning frameworks like Pytorch and Tensorflow have a Parameter datatype. Whenever a new type of layer is defined, both built-in and custom, you use this datatype to explicitly tell the framework which tensors should be treated as Parameters.

## Overview of how Optimization is done in a Neural Network
So when you build the network architecture, the model's parameters include the parameters of all the layers in that architecture. When training starts, you initialize those parameters with random values.

When you create the Optimizer you tell it the set of parameters that it is responsible for updating during training. In most cases this includes all the model's parameters. However there are many scenarios where you want to provide only a subset of the parameters for training. 

For instance, in a Generative Adversarial Network (GAN), the model has not one optimizer, but two. Each Optimizer manages only half of the parameters of the model. 

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Optim-1.png)
*An Optimizer that contains a list of all its parameters which is a subset of the model's parameters (Image by Author)*

Then, after the forward and backward passes, the Optimizer goes through all the parameters it is managing, and updates each one with an updated value based on:
- Parameter's current value
- Gradients for the parameter
- Learning Rate and other hyperparameter values

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Update-1.png)
*Parameter Update Formula showing new value = old value + hyperparam * delta (Image by Author)*

## Optimization Hyperparameters
All Optimizers require a Learning Rate hyperparameter. In addition, other hyperparameter depend on the specific optimization algorithm that you are using. For instance, momentum-based algorithms like Adam require a 'momentum' parameter. Other hyperparameters might include a 'beta' or 'weight decay'.

When you create an Optimizer you tell it what hyperparameter values you want it to use.

The hyperparameter values you choose have a big influence in how quickly the training happens as well as the model's performance based on your evaluation metrics. Therefore it is very important to choose these values well and to tune them for optimal results.

Since hyperparameters are so critical, neural networks give you a lot of fine-grained control over setting their values. Broadly speaking, there are two dimensions that you can control. The first of these involves Parameter Groups, which we will explore next. 

## Model Parameter Groups
Earlier we spoke about hyperparameters as though there was a single set of values for the whole network. But what if you want to choose different hyperparameters for different layers of the network?

Parameter Groups let you do just that. You can define multiple Parameter Groups for a network, each containing a subset of the model's layers.

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Param-2.png)
*A model consisting of layers, and different sections in different parameter groups (Image by Author)*

Now, using these you can choose different hyperparameter values for each Parameter Group. This is known as Differential Learning, because, effectively, different layers are 'learning at different rates'. 

## Differential Learning for Transfer Learning
A common use case where Differential Learning is applied is for Transfer Learning. Transfer Learning is a very popular technique in Computer Vision and NLP applications. Here, you take a large model that was pre-trained for, say, Image Classification with the ImageNet dataset and then repurpose it for a different much smaller set of images from your application domain.

When you do this you want to reuse all the pre-learned model parameters and only fine-tune it for your dataset. You do not want to relearn the parameters from scratch as that is very expensive.

In such a scenario, you typically split the network into two parameter groups. The first set consists of all the early CNN layers that extract image features. The second set consists of the last few Linear layers that acts as a Classifier for those features.

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Param-3.png)
*Transfer Learning showing a CNN section and a Linear Classifier section which are put into different Parameter Groups. (Image by Author)*

Now you can train the first parameter group using a very low learning rate so that the weights change very little. You can use a higher learning rate for the second parameter group so that the Classifier learns about the new domain images rather than the original set.

#### Differential Learning with Pytorch (and Keras - custom logic)
Pytorch Optimizer gives us a lot of flexibility in defining parameter groups and hyperparameters tailored for each group. This makes it very convenient to do Differential Learning.

Keras does not have built-in support for parameter groups. You have to write custom logic within your custom training loop to partition the model's parameters in this way with different hyperparameters.

We've just seen how to adjust hyperparameters using Parameter Groups. The second dimension to exercise fine-grained control on hyperparameter tuning involves the use of Schedulers.

## Hyperparameter Schedulers
So far we've talked about hyperparameters as though they were fixed values decided upfront before training. What if you wanted to vary the hyperparameter values over time as training progresses?

That is where Schedulers come in. They let you decide the hyperparameter value based on the training epoch. There are several standard Scheduler algorithms where you specify the start and end of a range of hyperparameter values. At each training iteration, the algorithm then calculates the hyperparameter value to use based on a mathematical formula.

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Scheduler-1.png)
*Show a network architecture with Scheduler next to the Optimizer. and a box showing LR or other hyperparam being modified by the Schduler(Image by Author)*

A Scheduler is considered a separate component and is an optional part of the model. If you don't use a Scheduler the default behavior is for the hyperparameter values to be constant throughout the training process. A Scheduler works alongside the Optimizer and is not part of the Optimizer itself.

Pytorch and Keras have several popular built-in Schedulers such as Exponential, Cosine and Cyclic Schedulers.

We have now seen all the different ways that hyperparameter values can be controlled during training. The simplest technique is to use a single fixed Learning Rate hyperparameter for the model. The most flexible is to vary the LR and other hyperparameters along both dimensions - different hyperparameter values for different layers, and simultaneously vary them over time over the course of the training cycle.

![]({{ site.baseurl }}/assets/images/OptimizerOverview/Scheduler-2.png)
*Show Param Groups on the vertical dimension and Schedulers on the horizontal, with training epoch and showing LR varying (Image by Author)*

#### Do not confuse this with how some Optimization Algorithms adjust Learning Rates
A quick clarification note - Do not confuse the above discussion with what some Optimization algorithms (like RMSProp) do - they also choose different learning rates for different parameters, based on the gradients of those parameters.

That is internal to some Optimization algorithms and is handled automatically by those algorithms. As the model designer it is invisible to you. The learning rate varies based on gradients and not based on the training epoch, as is the case with Schedulers. 

## Conclusion

## Quick Summary of popular Optimizers and Schedulers

## Optimization using Gradient Descent

Instead, it might be easier to imagine each of those millions of parameters independently. We can picture a 2D curve with only 1 parameter. Think of this as keeping all other parameters constant and varying only this single parameter and plotting the loss and gradients for it. Then imagine that there are millions of such separate curves.

Also, just to be clear, although the trajectory of the visualization shows you going 'down the slope' as you traverse the 3D curve, the optimizer itself operates only on the horizontal 2D parameter plane, and only indirectly moves the loss along the vertical dimension.

The optimizer updates the parameter value from one point to another on the horizontal axis. In turn, in the next iteration, the corresponding loss gets reduced along the vertical axis.

So, what we've just seen is that, depending on your starting point within the Loss Landscape, which you select randomly, gradient descent might encounter landscape features that are very hard to traverse.

## Obsolete

There are many types of optimizers that perform the final step above, with a goal of progressively reducing the loss in each iteration, till they find the minimum loss for the network.


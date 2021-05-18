---
layout: post
title: Neural Network Architectures Explained Visually - Batch Norm
subtitle: A Gentle Guide to the indispensable Batch Norm layer, in Plain English
imagecaption: Photo by [](https://unsplash.com/@hirmin) on [Unsplash](https://unsplash.com) 
categories: [ Neural, tutorial ]
author: ketan
tags: featured
image: https://images.unsplash.com/photo-1571156425562-12341e7c9aae?w=1200
---

## Why does Batch Norm work?
There is no dispute that Batch Norm works wonderfully well and provides substantial measurable benefits to deep learning architecture design and training. However, curiously, there is still no universally agreed answer about what gives it its amazing powers.

To be sure, many theories have been proposed. But over the years, there is disagreement about which of those theories is the right one.

The first explanation for why Batch Norm worked, by the original inventors was based on something called Internal Covariate Shift. Later in another [paper](https://arxiv.org/pdf/1805.11604.pdf) by MIT researchers, that theory was refuted, and an alternate proposed based on Smoothening of the Loss and Gradient curves. These are the two most well-known hypotheses, so let's go over these below.

#### Internal Covariate Shift
If you're like me, I'm sure you find this terminology quite intimidating! What does it mean, in simple language?

Let's say that we want to train a model and the ideal target output function (although we don't know it ahead of time) that the model needs to learn is as below.

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-1.png)
*Target function (Image by Author)*

Somehow, let's say that the training data values that we input to the model cover only a part of the range of output values. The model is therefore able to learn only a subset of the target function. 

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-2.png)
*Training data distribution (Image by Author)*

The model has no idea about the rest of the target curve. It could be anything.

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-3.png)
*Rest of the target curve (Image by Author)*

Suppose that we now feed the model some different testing data as below. This has a very different distribution from the data that the model was initially trained with. The model is not able to generalize its predictions for this new data.

For instance, this scenario can happen if we trained an image classification model, say, with pictures of passenger planes and then test it later with pictures of military aircraft.

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-4.png)
*Rest of the target curve (Image by Author)*

This is the problem of Covariate Shift - the model is fed data with a very different distribution than what it was previously trained with - _even though that new data still conforms to the same target function_. 

For the model to figure out how to adapt to this new data, it has to re-learn some of its target output function. This slows down the training process. Had we provided the model with a representative distribution that covered the full range of values from the beginning, it would have been able to learn the target output sooner.

Now that we understand what "Covariate Shift" is, let's see how it affects network training.

During training, each layer of the network learns an output function to fit its input. Let's say that during one iteration, Layer 'k' receives a mini-batch of activations from the previous layer. It then adjusts its output activations by updating its weights based on that input. 

However, in each iteration, the previous layer 'k-1' is also doing the same thing. It adjusts its output activations, effectively changing its distribution. 

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-5.png)
*How Covariate Shift affects Training (Image by Author)*

That is also the the input for layer 'k'. In other words, that layer receives input data that has a different distribution than before. It is now forced to learn to fit to this new input. As we can see, each layer ends up trying to learn from a constantly shifting input, thus taking longer to converge and slowing down the training.

Therefore the proposed hypothesis was that Batch Norm helps to stabilize these shifting distributions and thus speeds up training. 

#### Loss and Gradient Smoothening
The MIT paper published results that challenge the claim that addressing Covariate Shift is responsbile for Batch Norm's performance, and puts forward a different explanation.

In a typical neural network, the "loss landscape" is not a smooth convex surface. It is very bumpy with sharp cliffs and flat surfaces. This creates a challenge for gradient descent - because it could suddenly encounter an obstacle in what it thought was a promising direction to follow. To compensate for this, the learning rate is kept low so that we take only small steps in any direction.

![]({{ site.baseurl }}/assets/images/OptimizerTechniques/Landscape-1.png)
*A neural network loss landscape [(Source](https://arxiv.org/pdf/1712.09913.pdf), by permission of Hao Li)*

If you would like to read more about this, please see my [article](https://ketanhdoshi.github.io/Optimizer-Techniques/) on neural network Optimizers that explains this in more detail, and how different Optimizer algorithms have evolved to tackle these challenges.

What Batch Norm does is to smoothen the loss landscape substantially by changing the distribution of the network's weights. This means that gradient descent can confidently take a step in a direction knowing that it will not find abrupt disruptions along the way. It can thus take larger steps by using a bigger learning rate.

Although this paper's findings have not been challenged so far, it isn't clear whether they've been fully accepted as conclusive proof to close this debate.

## Advantages of Batch Norm
The huge benefit that Batch Norm provides is to allow the model to converge faster and speed up training. It makes the training less sensitive to how the weights are initialized and to precise tuning of hyperparameters.

Batch Norm lets you use higher learning rates. Without Batch Norm, learning rates have to be kept small to prevent large outlier gradients from affecting the gradient descent. Batch Norm helps to reduce the effect of these outliers.

Batch Norm also reduces the dependence of gradients on the initial weight values. Since weights are initialized randomly, outlier weight values in the early phases of training can distort gradients. Thus it takes longer for the network to converge. Batch Norm helps to dampen the effects of these outliers.

## When is Batch Norm not applicable
Batch Norm doesn't work well with smaller batch sizes. That results in too much noise in the mean and variance of each mini-batch. 

Batch Norm is not used with recurrent networks. Activations after each timestep have different distributions, making it impractical to apply Batch Norm to it.

## Conclusion
Hopefully, this gives you an understanding of .....

And finally, if you liked this article, you might also enjoy my other series on Transformers as well as Audio Deep Learning.

Let's keep learning!
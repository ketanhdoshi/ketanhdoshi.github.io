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

Batch Norm is an essential part of the toolkit of the modern deep learning practitioner. Soon after it was introduced in this [paper](https://arxiv.org/pdf/1502.03167.pdf), it was recognized as being transformational in creating deeper neural networks that could be trained faster.

Batch Norm is a neural network layer that is now commonly used in many architectures. It often gets added as part of a Linear or Convolutional block, and helps to stabilize the network during training.

In this article we will explore what Batch Norm is, why we need it and how it works. Before we talk about Batch Norm itself, let's go over some background to understand the motivation for Batch Norm.

## Normalizing Input Data
When inputting data to a deep learning model, it is standard practice to normalize the data to zero mean and unit variance. What does this mean and why do we do this?

Let's say the input data consists of several features x1, x2,...xn. Each feature might have a different range of values. For instance, values for feature x1 might range from 1 through 5, while values for feature x2 might range from 1000 to 99999.

So, for each feature separately, we take all the values in the dataset and compute the mean and the variance. And then normalize the values using the formula below. 

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-1.png)
*How we normalize (Image by Author)*

The picture shows the effect of normalizing data. This ensures that the all the feature values are now on the same scale.

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-2.png)
*What zero centered data looks like (Image by Author)*

To understand what happens without normalization, let's look at an example with just two features that are on drastically different scales. Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on different scales. Otherwise the large feature will simply drown out the small feature.

Then during gradient descent, in order to "move the needle" for the Loss, the network would have to make a large update to one weight compared to the other weight. In that case the loss landscape looks like a narrow ravine. This can cause the gradient descent trajectory to oscillate back and forth along one dimension, from one slope of the valley to the other, thus taking more steps to reach the minimum.

Instead, if the features are on the same scale, the loss landscape is more uniform like a bowl. Gradient descent can then proceed smoothly down to the minimum.

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-3.png)
*Normalized data helps the network converge faster (Image by Author)*

## Need for Batch Norm
Consider any of the hidden layers of a network. The activations from the previous layer are simply the inputs to this layer. The same logic that requires us to normalize the input for the first layer will also apply to each of these hidden layers.

![]({{ site.baseurl }}/assets/images/BatchNorm/Network-1.png)
*The inputs of each hidden layer are the activations from the previous layer, and must also be normalized (Image by Author)*

In other words, if we are able to somehow normalize the activations from each previous layer then the gradient descent will converge better during training. This is what the Batch Norm layer does.

## How Does Batch Norm work?
Batch Norm is just another network layer that gets inserted between a hidden layer and the next hidden layer. It's job is to take the outputs from the first hidden layer and normalize it before passing it on as the input of the next hidden layer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-1.png)
*Picture of linear, bias, activation for layer 1 going to layer 2 (Image by Author)*

A Batch Norm layer consists of:
- Two learnable parameters, beta and gamma.
- Two non-learnable parameters, mu and sigma, that are saved as part of the 'state' of the Batch Norm layer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-2.png)
*Parameters of a Batch Norm layer (Image by Author)*

These parameters are per each Batch Norm layer, so if we have, say, three hidden layers and three Batch Norm layers in the network, we would have three learnable beta parameters and three learnable gamma parameters for the three layers. Similarly we would have three moving average means and three moving average variances.

During training, we feed the network one mini-batch of data at a time. During the forward pass, each layer of the network processes that mini-batch of data. The Batch Norm layer process its data as follows:
- The activations 'a' from layer 'L - 1' are passed as input to the Batch Norm. There is one activation vector for each feature in the data.
- For each activation vector ai separately, calculate the mean and variance of all the values in the mini-batch.
- Calculate the normalized values for each activation feature vector using the corresponding mean and variance. These normalized values now have zero mean and unit variance.
- This step is the huge innovation introduced by Batch Norm that gives it its power. Unlike the input layer, which requires all normalized values to have zero mean and unit variance, Batch Norm allows its values to be shifted (to a different mean) and scaled (to a different variance). It does this by multiplying the normalized values by a factor, beta, and adding to it a factor, gamma. Note that this is an element-wise multiply not a matrix multiply.
- What makes this innovation ingenious is that these factors are not hyperparameters (ie. constants provided by the model designer) but are trainable parameters that are learned by the network. In other words each Batch Norm layer is able to optimally find the best factors for itself, and can thus shift and scale the normalized values to get the best predictions.
- In addition, Batch Norm also keeps a running count of the Exponential Moving Average of the mean (mu) and variance (sigma). During training it simply calculates this EMA but does not do anything with it. At the end of training it simply saves this value as part of the layer's state, for use during the Inference phase. We will return to this point a little later when we talk about Inference. The Moving Average calculation uses a scalar 'momentum' denoted by alpha below. This is a hyperparameter that is used only for Batch Norm moving averages and should not be confused with the momentum that is used in the Optimizer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-3.png)
*Calculations performed by BN layer with numbered steps (Image by Author)*

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-4.png)
*Shifting and Scaling the normalized values (Image by Author)*

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-5.png)
*Exponential Moving Average calculation (Image by Author)*

After the forward pass, we do the backward as normal. Gradients would be calculated and updates done for all layer weights, as well as for all beta and gamma parameters in the Batch Norm layers.

- Bias may be set to 0.
- Shapes

#### Order of placement of Batch Norm layer
There are two alternatives for where the Batch Norm layer is placed in the architecture - before and after activation. The original paper placed it before, although I think you will find both options frequently mentioned in the literature. Some say 'after' gives better results.

#### Batch Norm during Inference
As we discussed above, during Training, Batch Norm starts by calculating the mean and variance for a mini-batch. However, during Inference, we have a single sample not a mini-batch. How do we obtain the mean and variance in that case? 

Here is where the two Moving Average parameters come in - the ones that we calculated during training and saved with the model. Calculating the mean and variance for the full data would be very expensive as we would have to keep all those values in memory during training. Instead, the Moving Average acts as a good proxy for the mean and variance of the data. It is much more efficient because the calculation is incremental - we have to remember only the most recent Moving Average.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-6.png)
*Batch Norm calculation during Inference (Image by Author)*

## Why does Batch Norm work?
Even though there is no dispute that Batch Norm works wonderfully well and provides substantial measurable benefits to deep learning architecture design and training, curiously, there is still no universally agreed answer about what gives it its almost magical powers.

To be sure, many theories have been proposed. But over the years, there is disagreement about which of those theories is the right one.

The first explanation for why Batch Norm worked, by the original inventors was based on something called Internal Covariate Shift. Later in another [paper](https://arxiv.org/pdf/1805.11604.pdf) by MIT researchers, that theory was refuted, and an alternate proposed based on Smoothening of the Loss and Gradient curves. These are the two most well-known hypotheses, so let's go over these below.

#### Internal Covariate Shift
If you're like me, I'm sure you find this terminology quite intimidating! What does it mean, in simple language?

Let's say the ideal target output function for our model is as below.

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-1.png)
*Target function (Image by Author)*

However, the training data points that are input to the model are as below. The model is therefore able to learn only a subset of the target function. 

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-2.png)
*Training data distribution (Image by Author)*

The model has no idea about the rest of the target curve.

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-3.png)
*Rest of the target curve (Image by Author)*

Suppose we now feed the model some different data as below. This has a very different distribution from the data that the model was initially trained with. The model now has to figure out how to adapt to this new data, and perhaps re-learn some of its target output function. 

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-4.png)
*Rest of the target curve (Image by Author)*

This is the problem of Covariate Shift - the model is fed data with a very different distribution than what it was previously trained with - _even though that new data still conforms to the same target function_. This slows down the training process. Had we provided the model with a representative distribution that covered the full range of values from the beginning, it would have been able to learn the target output more quickly.

Now that we understand what "Covariate Shift" is, let's see how it affects network training.

During training, each layer of the network learns an output function to fit its input. Let's say that during one iteration Layer 'k' receives a mini-batch of activations from the previous layer. It then adjusts its output activations by updating its weights based on that input. 

However, in each iteration, the previous layer 'k-1' is also doing the same thing. It adjusts its output activations, effectively changing its distribution. 

![]({{ site.baseurl }}/assets/images/BatchNorm/ICS-5.png)
*How Covariate Shift affects Training (Image by Author)*

That is also the the input for layer 'k'. In other words, that layer receives input data that has a different distribution than before. It is now forced to learn to fit to this new input. As we can see, each layer ends up trying to learn from a constantly shifting input, thus taking longer to converge and slowing down the training.

#### Loss and Gradient Smoothening
The MIT paper published results that challenge the claim that addressing Covariate Shift is responsbile for Batch Norm's performance, and puts forward a different explanation.

In a typical neural network, the "loss landscape" is not a smooth convex surface. It is very bumpy with sharp cliffs and flat surfaces. This creates a challenge for gradient descent - because it could suddenly encounter an obstacle in what it thought was a promising direction to follow. To compensate for this, the learning rate is kept low so that we take only small steps in any direction.

![]({{ site.baseurl }}/assets/images/OptimizerTechniques/Landscape-1.png)
*A neural network loss landscape [(Source](https://arxiv.org/pdf/1712.09913.pdf), by permission of Hao Li)*

If you would like to read more about this, please see my article on neural network Optimizers that explains this in more detail, and how different Optimizer techniques have evolved to tackle these challenges.

What Batch Norm does is to smoothen the loss landscape substantially by changing the distribution of the network's weights. This means that gradient descent can confidently take a step in a direction knowing that it will not find abrupt disruptions along the way. It can thus take larger steps by using a bigger learning rate.

Although this paper's findings have not been challenged so far, it isn't clear whether they've been fully accepted as conclusive proof to close this debate.

## Advantages of Batch Norm
The huge benefit that Batch Norm provides is to allow the model to converge faster and speed up training. It makes the training less sensitive to how the weights are initialized and to precise tuning of hyperparameters.

Batch Norm lets you use higher learning rates. Without Batch Norm, learning rates have to be kept small to prevent large gradients due to outliers from affecting the gradient descent. Batch Norm helps to reduce the effect of these outliers.

Batch Norm also reduces the dependence of gradients on the initial weight values. Since weights are initialized randomly, outlier weight values in the early phases of training can distort gradients. Thus it takes longer for the network to converge. Batch Norm helps to dampen the effects of these outliers.

## When is Batch Norm not applicable
Batch Norm doesn't work well with smaller batch sizes. That results in too much noise in the mean and variance of each mini-batch. 

Batch Norm is not used with recurrent networks. Activations after each timestep have different distributions, making it impractical to apply Batch Norm to it.

## Conclusion
Hopefully, this gives you an understanding of .....

And finally, if you liked this article, you might also enjoy my other series on Transformers as well as Audio Deep Learning.

Let's keep learning!
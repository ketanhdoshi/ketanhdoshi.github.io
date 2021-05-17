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

Batch Norm is an essential part of the toolkit of the modern deep learning practitioner. Soon after it was introduced in the Batch Normalization [paper](https://arxiv.org/pdf/1502.03167.pdf), it was recognized as being transformational in creating deeper neural networks that could be trained faster.

Batch Norm is a neural network layer that is now commonly used in many architectures. It often gets added as part of a Linear or Convolutional block, and helps to stabilize the network during training.

In this article we will explore what Batch Norm is, why we need it and how it works. 

But before we talk about Batch Normalization itself, let's start with some background about Normalization.

## Normalizing Input Data
When inputting data to a deep learning model, it is standard practice to normalize the data to zero mean and unit variance. What does this mean and why do we do this?

Let's say the input data consists of several features x1, x2,...xn. Each feature might have a different range of values. For instance, values for feature x1 might range from 1 through 5, while values for feature x2 might range from 1000 to 99999.

So, for each feature column separately, we take the values of all samples in the dataset and compute the mean and the variance. And then normalize the values using the formula below. 

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-1.png)
*How we normalize (Image by Author)*

In the picture below, we can see the effect of normalizing data. The original values (in blue) are now centred around zero (in red). This ensures that all the feature values are now on the same scale.

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-2.png)
*What normalized data looks like (Image by Author)*

To understand what happens without normalization, let's look at an example with just two features that are on drastically different scales. Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on different scales. Otherwise the large feature will simply drown out the small feature.

Then during gradient descent, in order to "move the needle" for the Loss, the network would have to make a large update to one weight compared to the other weight. In that case the loss landscape (left picture below) looks like a narrow ravine. This can cause the gradient descent trajectory to oscillate back and forth along one dimension, from one slope of the valley to the other, thus taking more steps to reach the minimum.

Instead, if the features are on the same scale, the loss landscape is more uniform like a bowl (right picture below). Gradient descent can then proceed smoothly down to the minimum.

![]({{ site.baseurl }}/assets/images/BatchNorm/Normalize-3.png)
*Normalized data helps the network converge faster (Image by Author)*

## The need for Batch Norm
Now that we understand what Normalization is, the reason for needing Batch Normalization starts to become clear.

Consider any of the hidden layers of a network. The activations from the previous layer are simply the inputs to this layer. For instance, from the perspective of Layer 2 in the picture below, if we "blank out" all the previous layers, the activations coming from Layer 1 are no different from the original inputs.

The same logic that requires us to normalize the input for the first layer will also apply to each of these hidden layers.

![]({{ site.baseurl }}/assets/images/BatchNorm/Network-1.png)
*The inputs of each hidden layer are the activations from the previous layer, and must also be normalized (Image by Author)*

In other words, if we are able to somehow normalize the activations from each previous layer then the gradient descent will converge better during training. This is precisely what the Batch Norm layer does for us.

## How Does Batch Norm work?
Batch Norm is just another network layer that gets inserted between a hidden layer and the next hidden layer. It's job is to take the outputs from the first hidden layer and normalize it before passing it on as the input of the next hidden layer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-1.png)
*The Batch Norm layer normalizes activations from Layer 1 before they reach layer 2 (Image by Author)*

Just like the parameters (eg. weights, bias) of any network layer, a Batch Norm layer also has parameters of its own:
- Two learnable parameters, called beta and gamma.
- Two non-learnable parameters (Mean Moving Average and Variance Moving Average) that are saved as part of the 'state' of the Batch Norm layer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-2.png)
*Parameters of a Batch Norm layer (Image by Author)*

These parameters are per Batch Norm layer. So if we have, say, three hidden layers and three Batch Norm layers in the network, we would have three learnable beta and gamma parameters for the three layers. Similarly for the Moving Average parameters.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-5.png)
*Each Batch Norm layer has its own copy of the parameters (Image by Author)*

During training, we feed the network one mini-batch of data at a time. During the forward pass, each layer of the network processes that mini-batch of data. The Batch Norm layer process its data as follows:
1. The activations from the previous layer are passed as input to the Batch Norm. There is one activation vector for each feature in the data.
2. For each activation vector separately, calculate the mean and variance of all the values in the mini-batch.
3. Calculate the normalized values for each activation feature vector using the corresponding mean and variance. These normalized values now have zero mean and unit variance.
4. This step is the huge innovation introduced by Batch Norm that gives it its power. Unlike the input layer, which requires all normalized values to have zero mean and unit variance, Batch Norm allows its values to be shifted (to a different mean) and scaled (to a different variance). It does this by multiplying the normalized values by a factor, beta, and adding to it a factor, gamma. Note that this is an element-wise multiply not a matrix multiply.

   What makes this innovation ingenious is that these factors are not hyperparameters (ie. constants provided by the model designer) but are trainable parameters that are learned by the network. In other words each Batch Norm layer is able to optimally find the best factors for itself, and can thus shift and scale the normalized values to get the best predictions.
5. In addition, Batch Norm also keeps a running count of the Exponential Moving Average (EMA) of the mean and variance. During training it simply calculates this EMA but does not do anything with it. At the end of training it simply saves this value as part of the layer's state, for use during the Inference phase. We will return to this point a little later when we talk about Inference. The Moving Average calculation uses a scalar 'momentum' denoted by alpha below. This is a hyperparameter that is used only for Batch Norm moving averages and should not be confused with the momentum that is used in the Optimizer.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-3.png)
*Calculations performed by Batch Norm layer (Image by Author)*

Below, we can see the shapes of these vectors. The values that are involved in computing the vectors for a particular feature are also highlighted in red. However, remember that all feature vectors are computed in a single matrix operation.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-4.png)
*Shapes of Batch Norm vectors (Image by Author)*

After the forward pass, we do the backward pass as normal. Gradients would be calculated and updates done for all layer weights, as well as for all beta and gamma parameters in the Batch Norm layers.

#### Order of placement of Batch Norm layer
There are two opinions for where the Batch Norm layer should be placed in the architecture - before and after activation. The original paper placed it before, although I think you will find both options frequently mentioned in the literature. Some say 'after' gives better results.

![]({{ site.baseurl }}/assets/images/BatchNorm/Network-2.png)
*Batch Norm can be used before or after activation (Image by Author)*

#### Batch Norm during Inference
As we discussed above, during Training, Batch Norm starts by calculating the mean and variance for a mini-batch. However, during Inference, we have a single sample not a mini-batch. How do we obtain the mean and variance in that case? 

Here is where the two Moving Average parameters come in - the ones that we calculated during training and saved with the model. We use those saved mean and variance values for the Batch Norm during Inference.

![]({{ site.baseurl }}/assets/images/BatchNorm/BN-6.png)
*Batch Norm calculation during Inference (Image by Author)*

Ideally, during training, we could have calculated and saved the mean and variance for the full data. But that would be very expensive as we would have to keep values for the full dataset in memory during training. Instead, the Moving Average acts as a good proxy for the mean and variance of the data. It is much more efficient because the calculation is incremental - we have to remember only the most recent Moving Average.

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
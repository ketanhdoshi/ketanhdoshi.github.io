---
layout: post
title: Image Captions
categories: [ Jekyll, tutorial ]
image: https://miro.medium.com/max/1000/0*NlO0vViTFWhERLds
---

### A Gentle Guide to Image Captions, in Plain English

Image Captioning is a fascinating application of deep learning that has made tremendous progress in recent years. What makes it even more interesting is that it brings two together both Computer Vision and NLP.

What is Image Captioning? It takes an image as input and produces a short textual summary describing the content of the photo.

![By IDS.photos from Tiverton, UK - Labrador on Quantock, CC BY-SA 2.0, https://commons.wikimedia.org/w/index.php?curid=25739129 (Source https://en.wikipedia.org/wiki/Labrador_Retriever#/media/File:Labrador_on_Quantock_(2175262184).jpg)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Example-1.jpg)

eg. A relevant caption for this image might be "Dog standing in the grass" or "Labrador Retriever standing in the grass".

In this article my goal is to provide an overview of the techniques and architectures that are commonly used to tackle this problem. I will also go over an example demo application using one of these approaches and explain the detailed steps required to build it.

## High-level Approach for Image Captioning

Broadly speaking there are two primary components:

1. An Image Feature Processor
   This takes the source photo as input and produces an encoded representation of it that captures its essential features.

   ![Image input to an 'image Encoder' and outputs an encoded vector (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-1.jpg)

   This generally uses a CNN architecture, and is usually done using transfer learning. We take a CNN model that was pre-trained for image classification and remove the final section which is the 'classifier'. 
   
   The 'backbone' of this model consists of several CNN blocks which progressively extract various features from the photo, and generate a compact feature map that captures the most important elements in the picture. 
   
   eg. It starts by extracting simple geometric shapes like curves and semi-circles in the initial layers, progresses to higher-level structures such as noses, eyes and hands and eventually identifies elements such as faces and wheels.

   ![Image classification network showing the two sections and cutting off the classifier (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-2.jpg)

   In an Image Classification model, this feature map is then fed to the last stage which is the Classifier that generates the final output prediction of the class (eg. cat or car) of the primary object in the image.

   When applying this model for Image Captioning, we are interested in the feature map representation of the image, and do not need the classification prediction. So we keep the backbone and remove the classifier layers. 

2. A Text Sequence Generator
   This takes the encoded representation of the photo produced by the Image Feature Processor and outputs a sequence of words describing the photo.

   Typically this is a Recurrent Network model consisting of a stack of LSTM layers fed by an Embedding layer and ending in the Output layers.

   ![LSTM network with initial state and sequence so far as input, output fed back (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-3.jpg)
   
   It takes the image encoded vector as its initial state and is seeded with a minimal input sequence consisting of only a 'Start' token. It generates its prediction in a loop, outputting one word at a time which is then fed back to the network as input for the next iteration. Therefore at each step it takes the sequence of words predicted so far and generates the next word in the sequence. Finally it outputs an 'End' token which completes the sequence. That sequence is then output as the predicted caption.

3. Sentence Generator
   
   ![ (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-4.jpg)

Almost all Image Captioning architectures use this approach with the two components we've just seen. However, over the years many variations of this framework have evolved.

## Deep Learning Architecture - "Inject"
Perhaps the most common deep learning architecture for Image Captioning is called the "Inject" architecture and directly connects up the Image Feature Processor directly to the Text Sequence Generator as described above.

![Image Feature Encoder connected to Text Generator (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-5.jpg)

## Deep Learning Architecture - "Merge"
The Inject architecture was the original architecture for Image Captioning and is still very popular. However an alternative which gets called the "Merge" architecture has been found to produce better results.

Rather than connecting the Image Processor as the input of the Text Generator sequentially, the two components operate independently of each other. In other words, the CNN architecture processes only the image and the Text Generator processes only the text generated so far.

![Merge architecture (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-6.jpg)

The outputs of these two networks are then combined together with a simple Linear and Softmax layer, which does the job of interpreting those outputs and producing the final predicted caption.

This allows us to use transfer learning not just for the Image Processor but for the Text Generator as well. We can use a pre-trained language model to encode the text sequence.

Many different ways of combining the outputs have been tried eg. concatenate, multiplication and so on. The approach that usually works best is to use addition.

## Deep Learning Architecture - Object Detection backbone
Earlier we talked about using the backbone from a pre-trained Image Classification model for the Image Feature Processor. This type of model is usually trained to identify a single class for the whole picture. 

However, in most photos you are likely to have multiple objects of interest. Instead of using an Image Classification backbone, why not use a pre-trained Object Detection backbone?

![Object Detection backbone (By Comunidad de Software Libre Hackem [Research Group] - https://www.youtube.com/watch?v=ZmMFsL1ahI4, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=92687514)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-7.jpg)

Since the Object Detection model identifies bounding boxes around multiple prominent objects in the scene, it is able to provide a richer encoded representation of the image, which can then be used by the Text Generator to include a mention of all of those objects in its caption.

## Deep Learning Architecture - "Inject" with Attention
Over the last few years, the use of Attention with NLP models has been gaining a lot of, well, attention :-). It has been found to significantly improve performance of NLP applications. As the model generates each word of the output, Attention helps it focus on the words from the input sequence that are most relevant to that output word.

It is therefore not surprising to find that Attention has also been applied to Image Captioning resulting in state-of-the-art results.

![Image Caption architecture with Attention (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-8.jpg)

As the Text Sequence Generator produces each word of the caption, Attention is used to help it concentrate on the part of the image that is most relevant to the word it is generating.

![Example of Attention focusing on different parts of the image (Image by Author)]({{ site.baseurl }}/assets/images/ImageCaptionArch/Arch-9.jpg)

For instance, for the caption "man standing next to car", the model focuses on the man in the photo as it generates the word 'man' and then shifts its focus to the car when it reaches the word 'car', as you would expect.

## Deep Learning Architecture - Transformers
When talking about Attention, the current giant is undoubtedly the Transformer architecture. It revolves around Attention at its core and does not use the Recurrent Network which has been a NLP mainstay for years.

A few different variants of the Transformer architecture have been proposed to address the Image Captioning problem. One approach attempts to encode not just the individual objects in the photo but also their spatial relationships, as that is important in understanding the scene. For instance, knowing whether an object is under, behind or next to another object provides useful context in generating a caption.

### Dense Captioning

What it is, with an example. But article is already too long?

### Merge -> Multimodal
### Third component - Sentence Generation
### Beam Search
### Bleu Score

## Conclusion
With the advances made in Computer Vision and NLP, today's Image Captioning models are able to produce results that almost match human performance. When you input a photograph and get back a perfect human-readable caption, it almost feels like science fiction!

We have just explored the common approaches that are used to achieve this. We are now in a good position to look under the covers and get into the details. In my next article I will go through an example demo application step by step so we can see exactly how it works.
---
layout: post
title: Transformers Explained Visually -. Why they work so well.
subtitle: A Gentle Guide to why they way it computes Attention Boosts Performance, in Plain English.
imagecaption: Photo by [Joshua Sortino](https://unsplash.com/@sortino) on [Unsplash](https://unsplash.com)
categories: [ Transformers, tutorial ]
author: ketan
tags: featured
image: https://images.unsplash.com/photo-1488229297570-58520851e868?w=1200
---

Transformers have taken the world of NLP by storm in the last few years. Now they are being used with success in applications beyond NLP as well.

The Transformer gets its powers because of the Attention module. And this happens because it captures the relationships between each word in a sequence with every other word. 

But the key question is _how_ is it able to do that?

In this article, we will attempt to answer that question, and understand _why_ it performs the calculations that it does.

I have a few more articles in my series on Transformers. In those articles, we learned about the Transformer architecture, and walked through their operation during training and inference, step-by-step. We also explored under the hood and understood exactly how they work in detail. 

Our goal has been to understand not just how something works but why it works that way.

1. [**Overview of functionality**](https://ketanhdoshi.github.io/Transformers-Overview/) (_How Transformers are used, and why they are better than RNNs. Components of the architecture, and behavior during Training and Inference_)
2. [**How it works**](https://ketanhdoshi.github.io/Transformers-Arch/) (_Internal operation end-to-end. How data flows and what computations are performed, including matrix representations_)
3. [**Multi-head Attention**](https://ketanhdoshi.github.io/Transformers-Attention/) (_Inner workings of the Attention module throughout the Transformer_)
4. Why Attention Boosts Performance — this article (_Not just what Attention does but why it works so well. How does Attention capture the relationships between words in a sentence_)

To understand what makes the Transformer tick, we must focus on Attention. Let's start with the input that goes into it, and then look at how it processes that input.

## How does the input sequence reach the Attention module

The Attention module is present in every Encoder in the Encoder stack, as well as every Decoder in the Decoder stack. We'll zoom in on the Encoder attention first.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Encoder-1.png)
*Should we show only the Encoder part here not Decoder (Image by Author)*

As an example, let's say that we're working on a translation problem, where one sample source sequence in English is "The ball is blue". The target sequence in Spanish is "El bola es azul".

The source sequence is first passed through the Embedding and Position Encoding layer, which generates embedding vectors for each word in the sequence. The embedding is passed to the Encoder where it first reaches the Attention module.

Within Attention, the embedded sequence is passed through three Linear layers which produce three separate matrices - known as the Query, Key and Value. These are the three matrices that are used to compute the Attention Score.

The important thing to keep in mind is that each 'row' of these matrices corresponds to one word in the source sequence.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Data-1.png)
*(Image by Author)*

## Each input row is a word from the sequence

The way we will understand what is going on with Attention, is by starting with the individual words in the source sequence, and then following their path as they make their way through the Transformer. In particular we want to focus on what goes on inside the Attention Module.

That will help us clearly see how each word in the source and target sequences interacts with other words in the source and target sequences.

So as we go through this explanation, concentrate on what operations are being performed on each word, and how each vector maps to the original input word. We do not need to worry about many of the other details such as matrix shapes, specifics of the arithmetic calculations, multiple attention heads and so on, if they are not directly relevant to where each word is going.

So to simplify the explanation and the visualization, let's ignore the embedding dimension, and track just the rows for each word.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Data-2.png)
*(Image by Author)*

## Each word goes through a series of trainable transformations

Each such row has been generated from its corresponding source word by a series of transformations - embedding, position encoding, linear weights.

All of those transformations are trainable operations. This means that the weights used in those operations are not pre-decided but are learned by the model in such a way that they produce the desired output predictions. 

![]({{ site.baseurl }}/assets/images/TransformerWhy/Data-3.png)
*(Image by Author)*

The key question is, how does the Transformer figure out what set of weights will give it the best results? Keep this point in the back of your mind as we will come back to it a little later.

## Attention Score - Dot Product between Query and Key words

![]({{ site.baseurl }}/assets/images/TransformerArch/Attention-1.png)
*Multi-head attention (Image by Author)*

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-6.png)
*Attention Score formula with softmax, dk. Show Q, K, V not matrix drawings. (Image by Author)*

The first step within Attention is to do a matrix multiply (ie. dot product) between the Query (Q) matrix and a transpose of the Key (K) matrix. Watch what happens to each word.

We produce an intermediate matrix (let's call it a 'factor' matrix) where each cell is a matrix multiplication between two words. 

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-1.png)
*(Image by Author)*

For instance, each column in the fourth row corresponds to a matrix multiply between the fourth Query word with every Key word.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-2.png)
*(Image by Author)*

## Attention Score - Dot Product between Query-Key and Value words

The next step is a matrix multiply between this intermediate 'factor' matrix and the Value (V) matrix, to produce the attention score that is output by the attention module. Here we can see that the fourth row corresponds to the fourth Query word matrix-multiplied with all other Key and Value words.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-4.png)
*(Image by Author)*

This produces the Attention Score vector (Z) that is output by the Attention Module.

The way to think about the output score is that, for each word, it is the encoded value of every word from the "Value" matrix, weighted by the "factor" matrix. Where, the factor matrix is the dot product of the Query value for that specific word with the Key value of all words.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-7.png)
*Show the formula of Each word of value weighted by Factor Matrix (Image by Author)*

## Why do we need the Query, Key and Value words?

So the Query word can be interpreted as the word _for which_ we are calculating Attention. The Key and Value word is the word _to which_ we are paying attention ie. how relevant is that word to the Query word.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-5.png)
*Show a matrix with rows paying attention and columns being attention paid to(Image by Author)*

For example, for the sentence, "The ball is blue", the row for the word "blue" will contain the attention scores for "blue" with every other word. Here, "blue" is the Query word, and the other words are the "Key/Value".

There are other operations being performed such as a division and a softmax, but we can ignore them in this article. They just change the numeric values in the matrices but don't affect the position of each word row in the matrix. Nor do they involve any inter-word interactions.

## What does the dot product tell us about each word? Dot Product tells us the similarity between words
So we have seen that the attention score is capturing some interaction between a particular word, and every other word in the sentence, by doing a matrix multiply, and then adding them up. But how does the matrix multiply help the Transformer determine the relevance between two words?

?? But how does that interaction capture the relevance of one word to another? ?? 

To understand this, remember that the Query, Key and Value rows are actually vectors with an Embedding dimension. Let's zoom in on how the matrix multiplication between those vectors is calculated. 

![]({{ site.baseurl }}/assets/images/TransformerWhy/Attn-3.png)
*Two vectors doing a matrix multiply ie. a multiply between corresponding elements and a sum (Image by Author)*

When we do a dot product between two vectors, we multiply pairs of numbers and then sum them up.
- If the two paired numbers (eg. 'a' and 'd' above) are both positive or both negative, then the product will be positive. The product will increase the final summation.
- If one number is positive and other negative, then the product will be negative. The product will reduce the final summation.
- If the product is positive, the larger the two numbers, the more they contribute to the final summation.

![]({{ site.baseurl }}/assets/images/TransformerWhy/xxx-7.png)
*Two vectors doing a matrix multiply ie. a multiply between corresponding elements and a sum (Image by Author)*

This means that if the signs of the corresponding numbers in the two vectors are aligned, the final sum will be larger.

## How does the Transformer learn the relevance between words?
This idea applies to the Attention score as well. If the vectors for two words are more aligned, the attention score will be higher.

So what is the behavior we want?

We want the attention score to be high for two words that are relevant to each other in the sentence. And the score to be low for two words which are unrelated to one another.

For example, for the sentence, "The black cat drank the milk", the word "milk" is very relevant to "drank", perhaps slightly less relevant to "cat" and irrelevant to "black".
We want "milk" and "drank" to produce a high attention score. for "milk" and "cat" to produce a slightly lower score and for "milk" and "black", to produce a negligible score.

This is the output we want the model to learn to produce.

For this to happen, the word vectors for "milk" and "drank" must be aligned. The vectors for "milk" and "cat" will diverge somewhat. And they will be quite different for "milk" and "black".

Let's go back to the point we had kept at the back of our minds - how does the Transformer figure out what set of weights will give it the best results?

The word vectors are generated based on the word embeddings and the weights of the Linear layers. Therefore the Transformer can learn those embeddings, Linear weights and so on to produce the word vectors as required above.

In other words, it will learn those embeddings etc in such a way that if two words in a sentence are relevant to each other, then their word vectors will be aligned. And hence produce a higher attention score. For words that are not relevant to each other, the word vectors will not be aligned and will produce a lower attention score.

Therefore the embeddings for "milk" and "drank" will be very aligned and produce a high attention score. They will diverge somewhat for "milk" and "cat" to produce a slightly lower score and will be quite different for "milk" and "black", to produce a negligible score.

This then is the principle behind the attention module. 

## Summarizing - What makes the Transformer tick?

The dot product between the Query and Key computes the relevance between each pair of words. This relevance is then used as a "weight" to compute a weighted sum of all the words. That weighted sum is output as the Attention Score.

The Transformer learns embeddings etc, in such a way that words that are relevant to one another are more aligned.

This is one reason for introducing the three Linear layers and making three versions of the input sequence, for the Query, Key and Value. That gives the Attention Module some more parameters that it is able to learn to tune the creation of the word vectors.

## Encoder Self-Attention in the Transformer

Attention is used in the Transformer in three places:
- Self-attention in the Encoder — the source sequence pays attention to itself
- Self-attention in the Decoder — the target sequence pays attention to itself
- Encoder-Decoder-attention in the Decoder — the target sequence pays attention to the source sequence

![]({{ site.baseurl }}/assets/images/TransformerAttn/Attn-1.png)
*(Image by Author)*

In the Encoder Self Attention, we compute the relevance of each word in the source sentence to each other word in the source sentence. This happens in all the Encoders in the stack.

## Decoder Self-Attention in the Transformer

Most of what we've just seen in the Encoder Self Attention applies to Attention in the Decoder as well, with a few small but significant differences.

![]({{ site.baseurl }}/assets/images/TransformerArch/Attention-3.png)
*(Image by Author)*

In the Decoder Self Attention, we compute the relevance of each word in the target sentence to each other word in the target sentence.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Decoder-1.png)
*Decoder Self Attention (Image by Author)*

## Encoder-Decoder Attention in the Transformer

In the Encoder-Decoder Attention, the Query is obtained from the target sentence, and the Key/Value from the source sentence. Thus it computes the relevance of each word in the target sentence to each word in the source sentence.

![]({{ site.baseurl }}/assets/images/TransformerWhy/Decoder-2.png)
*Encoder Decoder Attention (Image by Author)*

## Conclusion
Hopefully, .....

And finally, if you are interested in NLP, you might also enjoy .

[State-of-the-Art Techniques](https://ketanhdoshi.github.io/Audio-Intro/)

[Reinforcement Learning Made Simple (Part 1): Intro to Basic Concepts and Terminology](https://ketanhdoshi.github.io/Reinforcement-Learning-Intro/)

Let's keep learning!

## Matrix Factorization in Recommender Systems
Recommendation Systems make use of this idea using a technique known as Matrix Factorization. To gain some intuition about why this is significant, let's look at an example.

Recommendation Systems recommend a product item to a user. They use a product item vector that represents the features of each product, and a user vector that represents the preferences of each user. For instance, if this is for a book, the product features might be about the genre of the book eg. one feature could indicate whether it is a mystery, another could indicate whether it is humorous. Similarly, the user vector contains features for the mystery and humor genre preferences of the user.

Now we do a matrix multiply between a user vector and a product vector to figure out a recommendation score for whether the user would like that product. If the product's mystery feature and the user's mystery preference are both positive (ie. user likes mysteries and the book is a mystery), or both negative (ie. user doesn't like mysteries and the book is not a mystery), then the matrix dot product will be high, and the product should be recommended. 

Conversely, if one is positive and other negative....

Similarly, the higher the value of the book's mystery feature (ie. it has a high mystery content) and the higher the value of the user's mystery preference (the user really loves mysteries), the higher the recommendation score.

![]({{ site.baseurl }}/assets/images/TransformerWhy/xxx-8.png)
*Matrix Factorization (Image by Author)*

How does the system decide what this set of features should be? And what are their values for each user and each product eg. "How humorous is this particular book"?

These things are not pre-decided and fed in as input to the system. Rather, we feed in data about some items that some users liked and didn't like. From that, the system learns what features are important, and what the values for each product and user should be. In such a way, so as to produce high recommendation score results for the items that those users liked, and low scores for the items that they didn't like.

## Matrix multiply for Attention (Version 1)
This idea applies to the Attention score as well. If the vectors for two words are more aligned, the attention score will be higher.

The word vectors are generated based on the word embeddings and the weights of the Linear layers. Therefore the Transformer can learn those embeddings, Linear weights and so on, in such a way that if two words in a sentence are relevant to each other, then their word vectors will be aligned. And hence produce a higher attention score. For words that are not relevant to each other, the word vectors will not be aligned and will produce a lower attention score.

This is one reason for introducing the three Linear layers and making three versions of the input sequence, for the Query, Key and Value. That gives the Attention Module some more parameters that it is able to learn to tune the creation of the word vectors.

The embeddings encode the "features" of each word eg. "bat" and "ball" will be very related to one another, so a feature could whether the word is a sports object. In reality, these features are really just a set of numbers and may not translate easily to a human understandable feature.

For example, for the sentence, "The black cat drank the milk", the word "milk" is very relevant to "drank", perhaps slightly less relevant to "cat" and irrelevant to "black". Therefore the embeddings for "milk" and "drank" will be very aligned and produce a high attention score. They will diverge somewhat for "milk" and "cat" to produce a slightly lower score and will be quite different for "milk" and "black", to produce a negligible score.

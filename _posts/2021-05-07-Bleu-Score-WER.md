---
layout: post
title: Foundations of NLP Made Simple - Bleu Score, WER
subtitle: A Gentle Guide to popular NLP metrics, in Plain English
imagecaption: Photo by [](https://unsplash.com/@hirmin) on [Unsplash](https://unsplash.com) 
categories: [ NLP, tutorial ]
author: ketan
tags: featured
image: https://images.unsplash.com/photo-1571156425562-12341e7c9aae?w=1200
---

Most NLP applications such as machine translation, chatbots, text summarization, and language models generate some text as their output. In addition applications like image captioning or automatic speech recognition (ie. Speech-to-Text) output text, even though they may not be considered pure NLP applications.

## How good is the predicted output?
The common problem when training these applications is how do we decide how 'good' that output is? 

With applications like, say, image classification where the predicted output can be compared unambiguously with the target output to decide whether the output is correct or not. However, the problem is much trickier with applications where the output is a sentence.

In such a case we don't always have one universally correct answer - we could have many correct answers. When translating a sentence from German to English, for instance, two different people may come up with two slightly different answers, both of which are completely correct.

eg. "The ball is blue" and "The ball has a blue color".

The problem is even harder with applications like image captioning or text summarization, where the range of possible answers is even larger.

![]({{ site.baseurl }}/assets/images/BleuScore/Caption-1.png)
*The same image can have many valid captions(Image by Author)*

In order to evaluate the performance of our model we need a quantitative metric to measure the quality of its predictions.

## NLP Metric
Over the years a number of different NLP metrics have been developed to tackle this problem. One of the most popular such metrics is called the Bleu Score.

It is far from perfect, and it has many drawbacks. But it is simple to compute and understand and has several compelling benefits. In spite of many alternatives, it continues to be one of the most frequently used metrics.

It is based on the idea that the closer the predicted sentence is to the human-generated target sentence, the better it is. 

Bleu Score are between 0 and 1. A score of 0.6 or 0.7 is considered the best you can achieve. Even two humans would likely come up with different sentence variants for a problem, and would rarely achieve a perfect match. For this reason, a score closer to 1 is unrealistic in practice, and should raise a flag that your model is overfitting.  

Before we explain how Bleu Score is calculated, let's understand two concepts first viz. N-grams and Precision.

## N-gram
An 'n-gram' is actually a widely used concept from regular text processing, and is not specific to NLP or Bleu Score. It is just a fancy way of describing "a set of 'n' consecutive words in a sentence".

For instance, in the sentence "The ball is blue", we could have n-grams such as:
- 1-gram (unigram): "The", "ball", ..
- 2-gram (bigram): "The ball", "ball is", "is blue"
- 3-gram (trigram): "The ball is", "ball is blue"
- 4-gram: "The ball is blue"

Note that the words in an n-gram are taken in order, so "blue is The ball" is not a valid 4-gram.

## Precision
This metric measures the number of words in the Predicted Sentence that also occur in the Target Sentence.

Let's say, that we have:

**Target Sentence**: He eats an apple
**Predicted Sentence**: He ate an apple

We would normally compute the Precision using the formula:

_Precision = Number of correct predicted words / Number of total predicted words_ = 3 / 4

But using Precision like this is not good enough. There are two cases that we still need to handle.

#### Repetition
The first issue is that this formula allows us to cheat. We could predict a sentence:

**Predicted Sentence**: He He He

and get a perfect Precision score = 3 / 3 = 1

#### Multiple Target Sentences
Secondly, as we've already discussed, there are many different correct ways to express the same sentence. To account for this, in many NLP models, we might be given multiple acceptable target sentences that capture many different variations.

We account for these two scenarios using a modified Precision formula which we'll call "Clipped Precision".

#### Clipped Precision
Let's go through an example to understand how it works.

Let's say, that we have the following sentences:

**Target Sentence 1**: He eats a sweet apple
**Target Sentence 2**: He is eating a tasty apple
**Predicted Sentence**: He He He eats tasty fruit

We now do two things differently:
- We compare each word from the predicted sentence with all of the target sentences. If the word matches any target sentence, it is considered to be correct.
- We limit the count for each correct word to the maximum number of times that word occurs in the Target Sentence. This helps to avoid the Repetition problem. This will become clearer below.

![]({{ site.baseurl }}/assets/images/BleuScore/Precision-5.png)
*Clipped Precision(Image by Author)*

For instance, the word "He" occurs only once in each Target Sentence. Therefore, even though "He" occurs thrice in the Predicted Sentence, we 'clip' the count to one, as that is the maximum count in any Target Sentence.

_Clipped Precision = Clipped number of correct predicted words / Number of total predicted words_ = 3 / 6

NB: For the rest of this article, we will just use "Precision" to mean "Clipped Precision". 

## How is it calculated?

Let's say we have a NLP model that produced a predicted sentence as below. For simplicity we will take just one Target Sentence, but as in the example above, the procedure is very similar with multiple Target Sentences.

**Target Sentence**: The guard arrived late because it was raining
**Predicted Sentence**: The watchman arrived late because of the rain

To calculate the Bleu Score:

![]({{ site.baseurl }}/assets/images/BleuScore/Bleu-1.png)
*Bleu Score(Image by Author)*

BLEU-1 is simply the unigram precision, 
BLEU-2 is the geometric average of unigram and bigram precision, 
BLEU-3 is the geometric average of unigram, bigram, and trigram precision and so on

![]({{ site.baseurl }}/assets/images/BleuScore/Bleu-3.png)
*Precision Scores(Image by Author)*

The first step is to compute Precisions scores for 1-grams through 4-grams.

**Precision 1-gram**
Precision 1-gram = Number of correct predicted 1-grams / Number of total predicted 1-grams = 5 / 8

![]({{ site.baseurl }}/assets/images/BleuScore/Precision-1.png)
*Precision(Image by Author)*

**Precision 2-gram**
Precision 2-gram = Number of correct predicted 2-grams / Number of total predicted 2-grams
Let's look at all the 2-grams in our predicted sentence:

![]({{ site.baseurl }}/assets/images/BleuScore/Precision-2.png)
*Precision 2-gram(Image by Author)*

So, Precision 2-gram = ?? / ??

**Precision 3-gram**
Similarly, Precision 3-gram = 

![]({{ site.baseurl }}/assets/images/BleuScore/Precision-3.png)
*Precision 3-gram(Image by Author)*

**Precision 4-gram**
And, Precision 4-gram = 

![]({{ site.baseurl }}/assets/images/BleuScore/Precision-4.png)
*Precision 4-gram(Image by Author)*

**Brevity Penalty**
The second step is to compute a 'Brevity Penalty'.

If you notice the Precision formulae above, we could have output a predicted sentence consisting of a single word like "The' or "late". For this, the 1-gram Precision would have been 1/1 = 1, indicating a perfect score. This is obviously misleading because it encourages the model to output fewer words and get a high score.

To offset this, the Brevity Penalty penalises sentences that are too short.

![]({{ site.baseurl }}/assets/images/BleuScore/Bleu-2.png)
*Brevity Penalty(Image by Author)*

where c is blah blah
and r is blah blah

It first computes: _predicted length / target length_ = _number of words in predicted sentence / number of words in target sentence_

We can see that if you predict very few words, this value will be low. Then

_Brevity Penalty = min (1, predicted length / target length)_

This ensures that the Brevity Penalty cannot be larger than 1, even if the predicted sentence is much longer than the target.

Finally, the formula for Bleu Score is


![]({{ site.baseurl }}/assets/images/BleuScore/Bleu-4.png)
*Bleu Score formula(Image by Author)*


## Implementing Bleu Score in Python
The nltk library, which is a very useful library for NLP functionality, provides an implementation of Bleu Score.

{% gist da29ba701004ecbe14821a472aa48107 %}
https://gist.github.com/ketanhdoshi/da29ba701004ecbe14821a472aa48107

Now that we know how Bleu Score works, there are a few more points that we should note.

## Bleu Score is computed on a corpus not individual sentences
Although we used examples of matching single sentences, Bleu Score is calculated by considering the text of the entire predicted corpus as a whole.

Therefore you cannot compute Bleu Score separately on each sentence in the corpus, and then average those scores in some way.

## Strengths of Bleu Score
The reason that Bleu Score is so popular is because it has several strengths:
- It is quick to calculate and easy to understand. 
- It corresponds with the way a human would evaluate the same text
- Importantly, it is language independent making it straightforward to apply to your NLP models. 
- It can be used when you have more than one ground truth sentence.
- It used very widely, which makes it easier to compare your results with other work.

## Weaknesses of Bleu Score
Inspite of its popularity, Bleu Score has some weaknesses:
- It does not consider the meaning of words. It is perfectly acceptable to a human to use a different word with the same meaning eg. Use "watchman" instead of "guard". But Bleu Score considers that an incorrect word
- It looks only for exact word matches. Sometimes a variant of the same word can be used eg. "rain" and "raining", but Bleu Score counts that as an error.
- It ignores the importance of words. With Bleu Score an incorrect word like "to" or "an" that is less relevant to the sentence is penalized just as heavily as a word that contributes significantly to the meaning of the sentence.
- It does not consider the order of words eg. The sentence "The guard arrived late because of the rain" and "The rain arrived late because of the guard" would get the same Bleu Score even though the latter is quite different.

## Conclusion


And finally, if you liked this article, you might also enjoy my other series on Transformers as well as Audio Deep Learning.

Let's keep learning!
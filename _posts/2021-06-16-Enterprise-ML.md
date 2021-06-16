---
layout: post
title: Enterprise ML
subtitle: A Gentle Guide to Enterprise, in Plain English
imagecaption: Photo by [](https://unsplash.com/@hirmin) on [Unsplash](https://unsplash.com) 
categories: [ Neural, tutorial ]
author: ketan
tags: featured
image: https://images.unsplash.com/photo-1571156425562-12341e7c9aae?w=1200
---

## What is Enterprise ML?
What does it take to deliver an machine learning (ML) application that provides real business value to your company? 

After you have proved the substantial benefit that ML can bring to the company, how do you expand that effort to additional use cases, and really start to fulfill the promise of ML?

And once you've done that how do you scale up ML across the organization and streamline the ML development and delivery process to standardize ML initiatives, share and reuse work and iterate quickly?

What are the best practices that some of the world's leading tech companies have adopted? 

Over a series of articles, my goal is to explore these questions and understand the challenges and learnings along the way.
- How does one train a ML model in the "real world", and how is that different from building a ML prototype "in the lab"?
- How do you take that model to production and keep it running at peak performance month after month?
- What infrastructure, system architecture and tooling has been put in place by companies that are at the forefront of this trend?
- How do you build data pipelines to extract value from the vast amount of data collected at your company and make it available for your ML and Analytics use cases?

In this article we will dive deeper into the first and crucial step on which all other steps depend, of building and training a ML model.

But before we do that, let's first set the context and get a high-level overview of an organization's overall ML journey.

## Maturing along the ML journey
Typically, most companies that are able to successfully leverage ML go through various stages of maturity.

Let's say a company has gathered a rich set of data and wants to make use of nachine learning (ML) so that it noticeably improves the experience of your customers or impacts your business operations in a major way.

![]({{ site.baseurl }}/assets/images/EnterpriseML/Journey-1.png)
*ML journey (Image by Author)*

- Identifies a problem and defines the business objectives. Starts R&D activity to develop an ML model.
- First model trained. Pilot application ready in production.
- Early stage - handful of models in production for a year or two
- Intermediate stage - several models in production for a variety of business scenarios across multiple departments
- Advanced - agile ML application development, standardized tools and processes for quick experimentation and delivery.

The reason that we are even discussing this topic of "real world ML" is that this is a hard road. So many companies get stuck at the very first step and are not able to extract tangible business value from their ML investments.

Now that we've talked about the long-term journey, let's narrow in on a single ML application along this path, and look at the steps involved end-to-end.

## ML Application Lifecycle and Roles
Delivering an ML application involves several tasks. Over the last few years a number of specialized roles have popped up in the industry to perform these tasks.

Note that this area is still evolving and that there is no standard terminology. Different companies probably have slightly different interpretations for the tasks as well as the roles. The boundary between these roles isn't sharply defined. In some cases, the same person might perform all of these responsibilities. 

However these concepts are starting to crystallize and it is still helpful to get a broad sense of the process.

![]({{ site.baseurl }}/assets/images/EnterpriseML/Lifecycle-1.png)
*ML Application Lifecycle (Image by Author)*

Data Scientist, Data Engineer, etc

As we just saw, the first stage in the application lifecycle, is to build and train the ML model. This is often the most "glamorous" and most technically challenging part of the project. Let's zoom in to see what is involved in building a ML model.

## How is training a "real world" ML model different from a "demo" ML project?
There are no shortage of resources, tutorials, online courses and projects on the Internet that cover every possible technical aspect of building a machine learning or deep learning model for a range of applications. However, the majority of them cover building ML models in a very controlled environment in the lab, so to speak.

How does that differ from what you will encounter in the "real world"? By far the biggest difference has to do with access to a labelled dataset. Lab projects invariably start with a carefully curated dataset that has already been prepared for you. The data is cleaned, and neatly labelled. The problem is neatly bounded because the fields in the dataset have been selected and scoped.

In a real project, on the other hand, you start with a blank slate. Preparing your dataset becomes one of the most time-consuming aspects of the project. You have to figure out answers to a number of questions: 

- What data sources are available?
- How do we query and extract the data?
- What fields do they contain?
- What data features will we use?
- How do we get labels?
- Is the data formatted properly?
- Are there missing values or garbage values?
- What data slices/segments should we use?
- How much training data do I need?
- How can I augment my data?

Secondly, in a lab project, the emphasis is often on squeezing out every last drop of accuracy for your model, or to get state-of-the-art results. Often, if you're competing on Kaggle, a difference on the metric score between 79.0345 and 79.0312 can mean hundreds of ranks on the leaderboard. In a real world project, even a percent or two improvement of the metric is probably far less crucial. The accuracy of your model is probably only one factor in the overall business outcome. It is often far more important to deliver a working solution quickly, with a noticeable customer improvement, get feedback and iterate quickly.

## Model Building and Training Workflow
Let's say that the problem to be solved and the business objective is clear, and you have a beginning hypothesis for how to tackle it. Typically, there are several steps involved in creating your ML model.

![]({{ site.baseurl }}/assets/images/EnterpriseML/Workflow-1.png)
*ML Model Workflow (Image by Author)*

- **Data Discovery**: You might start by browsing your data sources to discover the data set you want to use. Importantly you also need to identify what data will be used as the target labels.
- **Data Cleaning**: The data is probably messy and requires verification and cleaning. There might be missing or invalid values, outliers, duplicates and so on. Some fields might have useless values eg. A field like "Churn Reason" has a lot of values that simply say "Unknown". Some values may not be formatted correctly eg. numbers, dates and so on. If you're dealing with images, you might have blurry pictures, images of different sizes and resolutions, inadequate lighting, photos taken from inappropriate angles and so on.
- **Exploratory data analysis**: Look at the data distributions to identify patterns and relationships between the fields. You might identify seasonal trends or slice the data into relevant segments etc. 
- **Feature Engineering**: Derive new features by enriching some fields, performing aggregates or rollups, doing calculations by combining multiple fields and so on. For instance, you might use a date field to extract features for the number of days since the beginning of the month or year, whether it is a holiday etc.
- **Feature Selection**: Identify the features that are most useful to the model for predicting results. Remove features that add no value to the model.
- **Model Selection**: You might try out several different machine learning algorithms or deep learning architectures.
- **Hyperparameter Tuning**: For each model there are several hyperparameter values to be optimized.
- **Model Training**: Pick a model, select some data features, try some hyperparameters and train the model.
- **Model Evaluation**: Test the model against the validation data set. Track and compare the metrics for each model.
- **Inference**: After a promising potential model (along with features and hyperparams) has been identified, build the logic to make predictions on unseen data.
- **Repeat, repeat and repeat**: Change something, try a different idea and keep doing this many, many times. 

## Model Training Challenges
Building a ML model is hard. Unlike most software development projects, where you know how to solve the problem at hand, an ML project has a lot of uncertainty and unknowns. In the beginning, you don't know what the solution is, whether it is even feasible or how long it might take. Estimating and planning timelines is highly inaccurate.

The work is very researchy and iterative, and requires a lot of experimental trial and error.

Often, ML models are black-boxes. When a model fails to produce good results and metrics you might not be sure exactly why it failed. In many cases, the fix is to simply make some guesses, try something different and hope that it improves performance.

Development is usually done in Jupyter notebooks by a Data Scientist, or a team of Data Scientists. The model is trained using a static dump of your dataset in CSV or Excel files. Training is run on the developer's local laptop, or perhaps on a VM in the cloud. 

In other words, development of the model is fairly standalone and isolated from the company's application and data pipeline.

After several weeks or months of work, you finally solve the problem, and have a model that is performing well "in the lab", so to speak.

However, as we will see in the next phase, this is only a very small part of the way to your destination. And bigger challenges and pitfalls lie ahead.

Now that we know what we're including in the "Model Building" phase, we are ready to look at what the "Everything Else" phase is.

## Conclusion

This is a journey, and like all journeys, it begins with a first step, of building your ML model. In many ways, this is the most technically challenging and exciting part. However, as we'll later see, it is not this phase, but the next one that often trips up most projects and prevents them from seeing the light of day. 

Hopefully, this gives you an understanding of .....

And finally, if you liked this article, you might also enjoy my other series on Transformers as well as Audio Deep Learning.

Let's keep learning!



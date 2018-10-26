---
# Page settings
layout: default
keywords: linear regression example model
comments: false
permalink: /docs/regression

# Hero section
title: Regression Example
description: How to quickly construct linear/logistic regression models

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: /docs/inference-posterior-sampling/
    next: 
        content: Next Page
        url: /docs/particle-filter/
---

# Introduction
Linear and logistic regression are two simple classes of a problem where we can model a system output as the result of linear combinations of input variables and weights.
Linear regression describes the case where our output is a number, and logistic regression describes the case where our output is a discrete, two-valued variable.
Keanu has the ability to help you quickly and easily create regression models with input data.
Keanu also allows you to specify what form of regularization you would like (i.e. if you would like to use ridge regression or lasso regression).
By default, unregularised linear regression is used, but if you wish, you can specify Lasso or Ridge regression.

# Building a simple linear regression model
```java
//Define Model Parameters
double weight = 2.0;
double offset = 20.0;
int numberOfSamples = 100;

//Define random input data vertex by sampling from uniform probability distribution between 0 and 10
DoubleVertex xGenerator = new UniformVertex(new long[]{1, numberOfSamples}, 0, 10);

//Define the desired output vertices
DoubleVertex yMu = xGenerator.multiply(weight).plus(offset);

//Define a vertex for taking noisy readings of output data
DoubleVertex yGenerator = new GaussianVertex(yMu, 1.0);

//Sample input data and then sample the corresponding noisy output data
DoubleTensor xData = xGenerator.sample();
xGenerator.setAndCascade(xData);
DoubleTensor yData = yGenerator.sample();

//Create a simple linear regression model and fit it with our input and output data
RegressionModel regressionModel = RegressionModel.withTrainingData(xData, yData)
    .withRegularization(RegressionRegularization.NONE)
    .build();

//It is now possible to use regressionModel.predict(value) to get a prediction of the output given an input value.
```
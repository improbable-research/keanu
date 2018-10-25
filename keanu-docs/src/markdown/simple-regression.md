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
{% snippet SimpleLinearRegressionExample %}
```
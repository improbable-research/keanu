---
# Page settings
layout: default
keywords: autocorrelation convergence mixing
comments: false
permalink: /docs/autocorrelation/

# Hero section
title: Calculating Autocorrelation
description: How do you calculate the autocorrelation of your samples?

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/plates/'
    next:
        content: Next page
        url: '/docs/examples/save-and-load/'

---

## Autocorrelation

Autocorrelation is a useful statistic for assessing mixing of a Markov chain. Keanu provides a method of 
calculating autocorrelation on samples.

### Example

With a network defined, we can get the autocorrelation vertex A. The result is 
a tensor containing the autocorrelation at varying lags.
```java
{% snippet ScalarAutocorrelation %}
```

When the samples are tensors, we need to specify the tensor index on which to calculate the autocorrelation.
For example, if the sample shape is `[1,5]` we can evaluate the autocorrelation at index `[0,1]`.
```java
{% snippet TensorAutocorrelation %}
```
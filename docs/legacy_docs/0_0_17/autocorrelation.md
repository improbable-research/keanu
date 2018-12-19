---
# Page settings
layout: default
keywords: autocorrelation convergence mixing
comments: false
version: 0.0.17
permalink: /docs/0_0_17/autocorrelation/

# Hero section
title: Calculating Autocorrelation
description: How do you calculate the autocorrelation of your samples?

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_17/plates/'
    next:
        content: Next page
        url: '/docs/0_0_17/save-and-load/'

---

## Autocorrelation

Autocorrelation is a useful statistic for assessing mixing of a Markov chain. Keanu provides a method of 
calculating autocorrelation on samples.

### Example

With a network defined, we can get the autocorrelation vertex A. The result is 
a tensor containing the autocorrelation at varying lags.
```java
NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
    bayesNet,
    bayesNet.getLatentVertices(),
    100
);
DoubleTensor autocorrelation = posteriorSamples.getDoubleTensorSamples(A).getAutocorrelation();
```

When the samples are tensors, we need to specify the tensor index on which to calculate the autocorrelation.
For example, if the sample shape is `[1,5]` we can evaluate the autocorrelation at index `[0,1]`.
```java
NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
    bayesNet,
    bayesNet.getLatentVertices(),
    100
);
DoubleTensor autocorrelation = posteriorSamples.getDoubleTensorSamples(A).getAutocorrelation(0,1);
```

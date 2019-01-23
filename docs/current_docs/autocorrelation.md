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
        url: '/docs/save-and-load/'

---

## Autocorrelation

Autocorrelation is a useful statistic for assessing mixing of a Markov chain. Keanu provides a method of 
calculating autocorrelation on samples.

### Example

#### Java

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
DoubleTensor autocorrelation = posteriorSamples.getDoubleTensorSamples(A).getAutocorrelation(0, 1);
```

#### Python

It's also possible to calculate the autocorrelation of samples in Python.

```python
algo = MetropolisHastingsSampler()
posterior_samples = sample(net=bayes_net, sample_from=bayes_net.get_latent_vertices(),
                           sampling_algorithm=algo, draws=100)
vertex_samples = posterior_samples.get('a')
ac = stats.autocorrelation(vertex_samples)
```

When the samples are `ndarrays` the index on which to calculate the autocorrelation can be specified 
as a tuple.

```python
algo = MetropolisHastingsSampler()
posterior_samples = sample(net=bayes_net, sample_from=bayes_net.get_latent_vertices(),
                           sampling_algorithm=algo, draws=100)
vertex_samples = posterior_samples.get('a')
ac = stats.autocorrelation(vertex_samples, (0,1))
```

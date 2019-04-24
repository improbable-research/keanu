---
# Page settings
layout: default
keywords: map inference
comments: false
version: 0.0.23
permalink: /docs/0_0_23/inference-map/

# Hero section
title: Inference with Maximum A-Posteriori
description: How do you calculate the most probable values for random variables given you have some observations?

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_23/tensors/'
    next:
        content: Next page
        url: '/docs/0_0_23/inference-posterior-sampling'
---

## Model Fitting

Now that we've learnt to describe and build a model in [describing your model]({{ site.baseurl }}/docs/0_0_23/getting-started), we want to put 
it to use! Keanu enables you to calculate the **most probable values** of components of your model given certain conditions
or 'observations'. More formally, these are known as *posterior estimates*, and we are going to look at how we can obtain these
through an optimization method called Maximum A Posteriori (MAP).

Another technique for obtaining posterior estimates is through sampling. Read more about that on [posterior-sampling]({{ site.baseurl }}/docs/0_0_23/inference-posterior-sampling).
Sampling should be used when optimization techniques are not appropriate, e.g: you have discrete variables in your model.


## Maximum A Posteriori

We are going to learn how to calculate posterior estimates of your model. Or, using Keanu terminology, enabling you 
to calculate the most likely value of latent vertices inside your Bayesian network.

We are going to calculate posterior estimates using an Optimizer. Keanu has two types of Optimizer:
* Gradient Optimizer
* Non-Gradient Optimizer

From the names it's pretty obvious what the difference is. The Gradient Optimizer utilises gradients that are calculated
on your bayesian network using [Automatic-Differentiation](http://www.columbia.edu/~ahd2125/post/2015/12/5/) (AD), whereas the Non-Gradient Optimizer does not use gradients.
Automatic Differentiation may sound technical and scary, but don't worry, Keanu handles all the calculations for you! All
you have to focus on is describing your model.

"Which one do I use then?" I hear you ask. "How do I know if my model can make use of gradients?"

Well it depends on whether your latent variables (the ones we're going to find the most probable value of) are continuous
or discrete. Keanu does not support gradient optimisation on discrete latents. Still not sure? Don't worry, Keanu can analyse
your model and determine whether or not to use gradients for you. The next section will teach you how to use this.


### Optimizer

Let's say you've described the [thermometer model]({{ site.baseurl }}/docs/0_0_23/examples/thermometer) and want to run MAP but you're not sure
whether to use the Gradient or Non-Gradient Optimizer. You can use the following code to let Keanu decide which one to use.

```java
BayesianNetwork bayesNet = new BayesianNetwork(temperature.getConnectedGraph());
Optimizer optimizer = Keanu.Optimizer.of(bayesNet);
optimizer.maxAPosteriori();

double calculatedTemperature = temperature.getValue().scalar();
```

The Optimizer mutates the values of the graph while finding the most probable values and leaves the graph in its
most optimal state. Therefore, to find the most probable value of a vertex once, simply get the `value` of the vertex.

### Gradient Optimizer

This section will focus on the parameters available to you on the Gradient Optimizer. A builder is available
for the Gradient Optimizer that lets you change any combination of the default parameters. The snippet below demonstrates
how to use the builder to change all of the available parameters.

#### Java

```java
GradientOptimizer optimizer = Keanu.Optimizer.Gradient.builderFor(temperature.getConnectedGraph())
    .algorithm(ConjugateGradient.builder()
        .maxEvaluations(5000)
        .relativeThreshold(1e-8)
        .absoluteThreshold(1e-8)
        .build())
    .build();
optimizer.maxAPosteriori();

double calculatedTemperature = temperature.getValue().scalar();
```

* `bayesianNetwork` (required) - The Bayesian Network (model) to run the Optimizer on
* `maxEvaluations` (default: max int) - The max number of iterations the Optimizer goes through before terminating
* `relativeThreshold` (default: 1e-8) - If the delta in the log prob is less than the log prob times relativeThreshold, the Optimizer has converged
* `absoluteThreshold` (default: 1e-8) - If the delta in the log prob is less than absoluteThreshold, the Optimizer has converged

#### Python

```python
optimizer = GradientOptimizer(bayes_net, algorithm=ConjugateGradient(max_evaluations=5000,
                                                                     relative_threshold=1e-8,
                                                                     absolute_threshold=1e-8))
optimizer.max_a_posteriori()
calculated_temperature = model.temperature.get_value()
```

### Non-Gradient Optimizer

The Non-Gradient Optimizer implements the [BOBYQA algorithm](http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf), which stands for Bound Optimization BY Quadratic Approximation. 
It attempts to find the minimum of a black box objective function by forming a quadratic approximation in a trust region 
and finding the solution of the approximation.

This section will focus on the parameters available to you on the Non-Gradient Optimizer. A builder is available
for the Non-Gradient Optimizer that lets you change any combination of the default parameters. The snippet below demonstrates
how to use the builder to change all of the available parameters. 

#### Java

```java
OptimizerBounds temperatureBounds = new OptimizerBounds().addBound(temperature.getId(), -250., 250.0);
NonGradientOptimizer optimizer = Keanu.Optimizer.NonGradient.builderFor(temperature.getConnectedGraph())
    .algorithm(BOBYQA.builder()
        .maxEvaluations(5000)
        .boundsRange(100000)
        .optimizerBounds(temperatureBounds)
        .initialTrustRegionRadius(5.)
        .stoppingTrustRegionRadius(2e-8)
        .build())
    .build();
optimizer.maxAPosteriori();

double calculatedTemperature = temperature.getValue().scalar();
```

* `bayesianNetwork` (required) - The Bayesian Network (model) to run the Optimizer on
* `maxEvaluations` (default: max int) - The max number of iterations the Optimizer goes through before terminating
* `boundsRange` (default: positive infinity) - a bounding box of 'allowed values' that the Optimizer can try around the starting point
* `optimizerBounds` (default: no bounds) - a bounding box of 'allowed values' for a specific vertex 
* `initialTrustRegionRadius` (default: 10) - initial trust region radius (refer to BOBYQA paper for detail)
* `stoppingTrustRegionRadius` (default: 1e-8) - stopping trust region radius (refer to BOBYQA paper for detail) 

#### Python

```python
optimizer = NonGradientOptimizer(bayes_net, algorithm=BOBYQA(max_evaluations=5000,
                                                             bounds_range=100000.,
                                                             initial_trust_region_radius=5.,
                                                             stopping_trust_region_radius=2e-8))
optimizer.max_a_posteriori()
calculated_temperature = model.temperature.get_value()
```
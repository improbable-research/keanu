---
# Page settings
layout: default
keywords: overview
comments: false
permalink: /docs/overview/

# Hero section
title: Overview
description: Keanu allows you to build models that acknowledge uncertainty

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/installation/'
    next:
        content: Next page
        url: '/docs/getting-started/'


---
## Probabilistic programming

Probabilistic programming is programming that accounts for uncertainty. You can express what you don't know about a
program by describing the unknowns as probability distributions. 

Having uncertainty in a program is especially common
when the program is a model of some real world process. For example, if you were writing a program that tried to mimic 
a person driving through traffic, then you would need a way to express the uncertainty about how closely they are following
the car in front of them in the real world.

Note: the following code snippets are probabilistic programming pseudocode; we'll introduce Keanu syntax afterwards. 

For simplicity's sake, let's take a simple program with an input `x` and an output `y`.

```
x = 1
y = 2 * x
``` 

Here we see that `x` is 1 and if we run the program we will find that `y` is 2.
 
### Propagating uncertainty forward

However, what if we **don't know** what `x` is exactly but we **do know** that it's somewhere between 1 and 2?

With probabilistic programming we might be able to write:

```
x = 1 to 2 uniformly
y = 2 * x
```

Now that we've described the uncertainty around `x` we can make statements about `y` that ***include*** the uncertainty
of `x`. For example, we can calculate that `y` is somewhere between 2 and 4 given what we know about `x`. 

Propagating this uncertainty forward can be incredibly useful, especially when the program is significantly more complex and it's 
not immediately obvious how the uncertainty affects the program's output.

### Propagating uncertainty backwards

What if we observe something about the output and want to know what that means about our input? For example, if `y`
represented something in the real world that a human observed to be between 4 and 4.5 then what can we say about `x`?

```
x = 1 to 2 uniformly
y = 2 * x
y observed to be between 4.0 and 4.5 uniformly
```

Given our observation of `y` we could infer that `x` is between 2 and 2.25. Since we had a prior belief that
`x` was between 1 and 2 then x is ***most probably*** 2.

### Probabilistic programming with Keanu

In Keanu this program would look like:

```java
{% snippet Overview %}
```

Here we observed `y` varies by 0.5 and can be as high as 4.5 and as low as 4.0. If we used one of the
inference algorithms (e.g. [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)) in Keanu we would find `x` is ~2

## A real-world example: rain, sprinklers and wet grass

If the simple `x` and `y` example seems contrived, consider the wet grass example that you might be familiar with
(if you're not, take a look at the following article on [Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network)).

Let's assume that you have some prior belief that it has rained (20% chance) and some prior belief about a water 
sprinkler being on (1% chance if it has rained and 40% chance if it has not rained). If you observe a patch 
of wet grass near the water sprinkler then what is the probability that it rained?

Here is how you'd express this in Keanu.

```java
{% snippet Wetgrass %}
```

### Describing your problem

The wet grass example describes a problem as a Bayesian Network. It then takes samples from the
posterior distribution, using:
 
 ```java
 MetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(...)
 ```
 
in order to determine the probability that it rained given the grass was observed
to be wet. 

Converting your problem to a dependency graph, which with prior beliefs makes this a 
[Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network), is the first step to applying Keanu 
to your problem. Once you have described your problem as a Bayesian Network, there are several potentially valuable questions
you can ask:
 
1. ***What is the most probable network state given my observations? (Maximum a posteriori)***

    This finds the most probable unknown values given what ***is*** known. If your probability distribution is differentiable,
    the gradient optimizer in Keanu will automatically calculate a gradient in order to efficiently find the 
    values for the unknowns that most closely match the observations. If your problem is not differentiable, 
    meaning it uses discrete events, has undefined areas or has especially not well behaved operations, then 
    you can either use a sampling algorithm or the non gradient based optimizer.  

1. ***What is the probability of an event given my observations? (Posterior sampling)***

    This uses one of the many sampling algorithms to sample from the posterior distribution, which
    is the distribution that describes probabilities given some observations.
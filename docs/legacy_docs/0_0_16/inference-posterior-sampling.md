---
# Page settings
layout: default
keywords: map inference
comments: false
version: 0.0.16
permalink: /docs/0_0_16/inference-posterior-sampling/

# Hero section
title: Inference by sampling
description: How do you ask probabilistic questions about your model given you have some observations?

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_16/inference-map/'
    next:
        content: Next page
        url: '/docs/0_0_16/regression'
---

## MCMC

Posterior distributions can become extremely complex. They are often high dimensional. 
As we make observations in our model, their shape shifts and changes accordingly.

If we know the form of a posterior distribution, we gain valuable information about our hidden (latent) variables.
Unfortunately, there's no easy way to perfectly determine the form of a posterior distribution, so we have
to rely on approximation.

Luckily, there are sampling algorithms that are designed to efficiently explore distributions and enable us
to gain insight into their form. 
[MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) stands for Markov Chain Monte Carlo, and describes a class of algorithms that help us do exactly that.
In this section we will be exploring a few MCMC algorithms.

### Metropolis Hastings

#### Algorithm

[Metropolis Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is the simplest
of the sampling algorithms we have implemented in Keanu, and it is a good place to start.

At each iteration of the algorithm, we do three things:

* Pick one of the vertices in the network and propose a new value for it
* Calculate the new likelihood of the network occuring in that state
* Either accept the value and use it as a new sample or reject it and use the previously accepted value as a sample instead

The Metropolis Hasting algorithm accepts or rejects proposed samples in such a way that as we keep sampling, 
our approximation gets closer to the true posterior distribution.

So how can we use Metropolis Hastings to sample from a distribution in Keanu?

#### Example

We define two normally distributed variables, A and B, that are centered around 20.0 with a sigma of 1.0.
This is an expression of our prior belief that A and B both have values around 20.0.

```java
DoubleVertex A = new GaussianVertex(20.0, 1.0);
DoubleVertex B = new GaussianVertex(20.0, 1.0);
```

We then receive a message from a researcher saying they've measured that A + B is 43.
However, the researcher is not entirely sure of this, so we create another normal distribution which is centered
around the sum of A and B with a sigma of 1.0.

We then observe this to be 43.0.

```java
DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
C.observe(43.0);
```

By making this observation and combining our 'belief' with our knowledge, we have shifted the distributions of A and B.

Let's now sample from this network to learn how the distributions of A and B have changed.

First we create a Bayesian Network from our distributions and find a non-zero start state.

```java
// Set the values of A and B so the Metropolis Hastings
// algorithm has a non-zero start state.
// Alternatively you could infer a possible start state
// using either a particle filter or something like
// bayesNet.probeForNonZeroProbability(100);
A.setValue(20.0);
B.setValue(20.0);

BayesianNetwork bayesNet = new BayesianNetwork(C.getConnectedGraph());
```

Now let's use Metropolis Hastings to sample from this network.

Metropolis Hastings accepts the following arguments:

* The network to sample from
* The vertices to return samples for (latent vertices)
* The number of samples to take

We will be taking 100,000 samples from the distributions of A and B.

```java
NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
        bayesNet,
        bayesNet.getLatentVertices(),
        100000
);
```

Now we have some samples, we can use the following code to take the average value across all the samples of A and B,
which will tell us their 'most likely value'.

```java
double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar(); //21.0
double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar(); //21.0

double actual = averagePosteriorA + averagePosteriorB; //42.0
```

As we can see, our prior belief and observations have combined and we've gained insight through sampling.

### Hamiltonian Monte Carlo

#### Algorithm

Hamiltonian Monte Carlo (HMC) brings two new concepts to Metropolis Hastings: Hamiltonian physics and gradient calculations.

Hamiltonian physics introduces the idea of a position and a momentum for a sample. Put simply, by knowing the gradient, we
can understand if the distribution is steep or shallow at our current position. We can then use Hamiltonian physics to move 
our samples around using this information.

This is crucial to understand in order to tweak the parameters of HMC, it's very sensitive to two parameters in particular:

* Leapfrog count
* Step size

A leapfrog is how a sample moves around the distribution. 
It's where the position and momentum of a sample are updated for a timestep.
The step size is how much time will pass between timesteps. 
The larger the stepsize, the more the position will change during each timestep.
The leapfrog count determines how many leapfrogs we will take in one sample. 
If we have a larger leapfrog count, the position will change a lot more than if we had a smaller leapfrog count.

If your stepsize and leapfrog parameters are not high enough, then you will not be able to effectively explore the distribution in time.
However, if your stepsize or leapfrog count are too high, then you will
move too large a distance each timestep and not gain any precise information about the shape of the distribution.

The trick is to find the largest step size and the smallest leapfrog count that return an effective sampling of the distribution.

But how does that look in practice? Well let's take an example - let's say you set the following parameters:

* Leapfrog count: 20
* Step size: 0.1

At each sample you have made 20 leapfrogs at a distance of 0.1 each. 
Therefore, you have covered a total distance of 2 before the likelihood is calculated. 
If your problem domain is a simple Gaussian that has a sigma of 1, then this will obviously be a problem as
we will be covering too large a distance and will never explore the Gaussian effectively.

If you are not receiving accurate samples, try lowering your step size to explore the space in finer detail.

The parameters are:

* The Bayesian network to sample from
* The vertices in the network to return samples for (latent vertices)
* The number of samples to take

```java
NetworkSamples posteriorSamples = Hamiltonian.withDefaultConfig().getPosteriorSamples(
        bayesNet,
        bayesNet.getLatentVertices(),
        2000
);
```

### NUTS

#### Algorithm

NUTS is built on top of HMC. 
As the sample moves through the distribution, there are features in place that stop it from performing 
a U-turn and re-exploring locations. 
Hence why it's called the 'No-U-Turn Sampler'. 
It also attempts to calculate and auto-tune those difficult leapfrog and step size parameters we encountered in HMC.

```java
NetworkSamples posteriorSamples = NUTS.withDefaultConfig().getPosteriorSamples(
        bayesNet,
        bayesNet.getLatentVertices(),
        2000
);
```

Let's explain the auto-tuning and target acceptance probability parameters a bit further.

Picking a step size in HMC is very challenging and can often result in you not exploring the distribution effectively.
NUTS calculates a step size for you based on your Bayesian network and then will continue to adapt it for a given
number of samples. 
It does this by using the target acceptance probability. 
To find an effective starting value of step size, the algorithm chooses a starting point and then makes many jumps of differing sizes, recording the likelihood of the attempted position. 
This is used to find the *largest* stepsize with a likelihood above that of the target acceptance probability.

However this initial step size may only be appropriate for the starting position. To account for this, the step size
will continue to be adapted to ensure it best explores the distribution for a specified amount of samples.
We recommend to adapt for 10% of the sample count.
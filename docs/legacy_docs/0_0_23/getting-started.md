---
# Page settings
layout: default
keywords: Changing The Graph
comments: false
version: 0.0.23
permalink: /docs/0_0_23/getting-started/

# Hero section
title: Getting Started
description: Learn how to build a model as a collection of random variables, observations and deterministic operations.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_23/overview/'
    next:
        content: Next page
        url: '/docs/0_0_23/data-io/'


---

## Your model as a Bayesian Network

You need to describe your model to Keanu as a Bayesian network. A network is built from vertices. 
Vertices represent variables, which may be random or deterministic, and edges represent dependencies between variables. 
Your model's state (i.e. data) is housed in these vertices as the vertex's `value`. 
The value of a vertex can depend on the value of a parent vertex and can be updated in one of
two ways.

Let's look at an example of two vertices A and B that contain some numbers as their values. Numbers from A and B are 
added together, which yields C.

```
(A) _
     \
      + -> (C)
(B) _/
```

If the number in A changes then the number in C will change as well and likewise for changes from B.

You can describe this in Keanu as:

```java
DoubleVertex A = new GaussianVertex(0, 1);
DoubleVertex B = new GaussianVertex(0, 1);
DoubleVertex C = A.plus(B);
```

### Propagating changes forward

If you change A, you can tell C to recalculate based off of A's new value and B's unchanged value. 
To do this: 

```java
A.setAndCascade(1.234);
```

### Evaluating upstream changes

But if you want to change both A and B then you probably don't want to have C update twice. In that
case you would prefer to calculate C after both and A and B have changed and therefore calculating 
C once. This can be done by:

```java
A.setValue(1.234);
B.setValue(4.321);
C.lazyEval();
```

or

```java
A.setValue(1.234);
B.setValue(4.321);
VertexValuePropagation.cascadeUpdate(A, B);
```

### Observing a value

Another central concept to Bayesian networks is observations. The value of a vertex can be "observed", which
effectively locks the value of the vertex. Observing a vertex raises a flag on the vertex that tells an
inference algorithm to treat the vertex in a special way.

Observing vertices that contain numbers is a special case and is described more in the docs on [Double vertices]({{ site.baseurl }}/docs/0_0_23/vertex-summary).
In the interest of keeping this simple, take for example the case where instead of multiplying A and B, we apply the logical AND operator to their values.

```
(A) _
     \
     AND -> (C)
(B) _/
```

In this example, A and B contain boolean values and C is true only if both A and B are true. To describe this network in Keanu:

```java
BooleanVertex A = new BernoulliVertex(0.5);
BooleanVertex B = new BernoulliVertex(0.5);
BooleanVertex C = A.and(B);
```

To observe that C is true:

```java
C.observe(true);
```

Now you can infer that A and B are also both true by sampling from the posterior distribution. Note: we will be covering MCMC sampling in the [Posterior Sampling]({{ site.baseurl }}/docs/0_0_23/inference-posterior-sampling/) section.

```java
A.observe(true);
B.observe(true);

KeanuProbabilisticModel model = new KeanuProbabilisticModel(C.getConnectedGraph());
NetworkSamples posteriorSamples = Keanu.Sampling.MetropolisHastings.withDefaultConfig().getPosteriorSamples(
    model,
    Arrays.asList(A, B),
    100000
).drop(10000).downSample(2);
double probabilityOfA = posteriorSamples.get(A).probability(isTrue -> isTrue.scalar() == true);
//probabilityOfA evaluates to 1.0
```
**You may be wondering why we go to all the hassle of doing inference** rather than just writing something like the following:
```java
//WRONG
A.lazyEval();
B.lazyEval();
System.out.println(A.getValue().scalar());
```
The issue here is that A and B are vertices in the computation graph that describes our prior and taking the value from A and B will just return a random value as if the BernoulliVertex was referenced in isolation. 
In this very contrived example it seems obvious to us how the value of C should propagate values to A and B but this is not always so straightforward.
In general, this process is known as *variable elimination* and it is not supported by Keanu. 
Therefore, in order to infer the values of A and B, you have to perform inference using a posterior sampling algorithm like [MCMC]({{ site.baseurl }}/docs/0_0_23/inference-posterior-sampling/).

**You may also be wondering why we also observe A and B to be true** in the above code. 
This is so that when our sampling algorithm (MCMC) starts to sample from the posterior, it will start from a network with a probability that is non-zero.
If we do not include this, then our network will get a random starting value, e.g. A is false and B is true, and then will discover that it is actually not possible to be in this state and will throw an error. 

In general, A and B are known as *latent variables* because we do not directly observe them. In more complex cases, we may not know what starting state (like A: true, B: true in this case) to use. 
There are a couple of techniques that solve this problem so that we can leverage Bayesian Inference.
Firstly, you might choose to use the `ParticleFilter` class in order to find the most probable state to start your algorithm from.
Alternatively, you can create a new Bayesian network and use this to probe some random configurations a certain number of times to see if it can find a possible one. 
```java
BayesianNetwork bayesianNetwork = new BayesianNetwork(C.getConnectedGraph());
bayesianNetwork.probeForNonZeroProbability(10);
```
Now you can run MCMC on the BayesNet as it will start off in the correct configuration.

Instead of running MCMC you could also run one of our [inference algorithms]({{ site.baseurl }}/docs/0_0_23/inference-map).
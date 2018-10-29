---
# Page settings
layout: default
keywords: lorenz example
comments: false
permalink: /docs/examples/lorenz/

# Hero section
title: Lorenz Example
description: A more complex example with a chaotic system

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/examples/thermometer/'
---

## Overview

This example aims to demonstrate two things:

* using Keanu with [Kotlin](https://kotlinlang.org/)
* how to solve a time-stepped problem

## The problem to model

Imagine you are watching the point below moving through space:

![Trajectory through phase space](https://upload.wikimedia.org/wikipedia/commons/1/13/A_Trajectory_Through_Phase_Space_in_a_Lorenz_Attractor.gif)

You'd begin to notice it has an interesting repeated pattern of movement. If you had a really
keen eye, you'd notice that its movement can be described by the
[Lorenz equations](https://en.wikipedia.org/wiki/Lorenz_system).

The Lorenz equations are particuarly hard to model as they describe a chaotic system. Two initial
starting states, no matter how close, will diverge almost immediately. 

In this example, you'll make observations on the point above and build a probabilistic model
using Keanu that can accurately describe its chaotic motion. To make things harder, you are going to
only observe the X coordinate, and to observe a noisy value of the X coordinate.

### How to model the problem

To start with, you'll need to build a simple program that can step through time and calculate
coordinates using the Lorenz equations. This will serve as the "Real World" system that you
can take observations of the X coordinate from, and compare the calculated coordinates of the
probabilistic model against.

Then you'll step through time, taking observations of the X coordinate from this "Real World" system, 
and feeding this value into the probabilistic model.

What does this probabilistic model look like? Fortunately, you already have a conceptual model of the
system and you know it follows movement described by the Lorenz equations: 

![Lorenz equations](https://wikimedia.org/api/rest_v1/media/math/render/svg/5f993e17e16f1c3ea4ad7031353c61164a226bb8)

Rather than performing a single timestep and then calculating the most likely value of the coordinates, 
you'll perform several timesteps at once and then calculate the most likely value. 

Let's define 10 timesteps as a window, and calculate the probable values at the end of every window.

## Pseudocode

At each timestep, you need to:

1. Increment the "Real World" system and take a reading of its coordinates.
2. Create a probabilistic graph that describes the Lorenz equations for that timestep.
3. Observe the X coordinate of this graph to be the X coordinate from the "Real World" system.

The pseudocode for a window looks like:

1. Run 10 timesteps
1. Calculate the most probable value of the X, Y and Z coordinates from the connected graph.
2. Use these probable values as a prior value of X, Y and Z in the graph for the next window.
3. Evaluate how close the probable values are to the actual value of the coordinates.

## Implementation

Let's start by calculating the coordinates of the "Real World" system, for a given number of timesteps.
We've prepared this ahead of time, so you can use our class, `LorenzModel`.

```kotlin
{% snippet LorenzModel %}
```

Let's define a starting point for the X, Y and Z coordinates. 
You have no prior knowledge of what the starting coordinates for the model will be, so let's define each
coordinate as a Gaussian around the origin.

```kotlin
{% snippet LorenzStartingPoint %}
```

You then want to iterate through these timesteps until either:

1. The calculation of the most probable values of X, Y and Z are within an acceptable tolerance
of the actual coordinates.
2. We reach our maximum timestep.

Place all of the following code inside a loop that describes these conditons:

```kotlin
{% snippet LorenzIterate %}
```

Let's now define a function that can calculate the next timestep of a Lorenz equation in Keanu
given these starting coordinates.

Fortunately, Kotlin supports operator overloading, so this will look exactly like a normal
program even though you're dealing with objects representing probability distributions.

```kotlin
{% snippet LorenzTimestep %}
```

Let's define a function that can compute the 10 timesteps in a window using the function you defined above. 

```kotlin
{% snippet LorenzWindow %}
```
 
You can now apply the observations of the X coordinate from the "Real World" system to each timestep
in the window calculated above. 

Let's define a function that does this.

```kotlin
{% snippet LorenzObservations %}
```

This function is iterating through each of the timesteps you calculated above, extracting the Gaussian
distribution that describes the X coordinate, and observing it to be the value taken from the
"Real World" system.

Brilliant, now let's calculate the most likely value for the X, Y and Z coordinates.

Hopefully by now the code for performing Gradient Optimisation will look familiar to you:

```kotlin
{% snippet LorenzOptimise %}
```

Now you've calculated the most probable values for the graph, you need to feed these into the next window.
As you're performing 10 timesteps in a window, you want to take the value of the 10th timestep as the
starting point of the next window.

Let's define a function to do that:

```kotlin
{% snippet LorenzGetWindowVals %}
```

Great, now you need to feed these back into the next window.

If you'll recall, at the start of the window, you defined the initial coordinates of the graph as:

```kotlin
{% snippet LorenzStartValues %}
```

Therefore, let's recalculate these Gaussians to include the calculated value of the most probable coordinates:

```kotlin
{% snippet LorenzNewStartValues %}
```

## Results

Can you have a guess at what you think the results will be before you run the code?

We are performing 10 timesteps in each window: how many windows do you think it will take before the
error between the probabilistic model and the "Real World" model is negligible?

Printing out the error of each window produces the following results.

```
Error: 4.528976372510882
Error: 3.72098313476314
Error: 2.665636788694244
Error: 1.812040602844943
Error: 1.2950274565457909
Error: 1.040466635607921
Error: 0.7338159078803722
Error: 0.38109769602303945
Error: 0.1774717811757308
Error: 0.11441697844622131
Error: 0.0918083820220631
Error: 0.06013662151356879
Error: 0.030568642144244644
Error: 0.02316773884356288
Error: 0.015850612891648755
Error: 0.008663664263278796
```

It takes 16 windows before the probabilistic model has converged on the "Real World" system, from only an
observation of the X coordinate!

## Code

Here is the completed code if you'd like to run it yourself.

```kotlin
{% snippet LorenzFull %}
```

## Java Code

Here is the same implementation, but done in Java.

```java
{% snippet LorenzJavaFull %}
```
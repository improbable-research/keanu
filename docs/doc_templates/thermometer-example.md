---
# Page settings
layout: default
keywords: example thermometer
comments: false
permalink: /docs/examples/thermometer/

# Hero section
title: Thermometer Example
description: You have two inaccurate thermometers and want to estimate the temperature of a room.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/save-and-load/'
    next:
        content: Next page
        url: '/docs/examples/lorenz/'

---

## Overview

Here's a simple example to demonstrate Keanu's syntax and how to solve a basic problem.

## The problem?

You are sitting in a room. Feeling rather hot you decide to measure the temperature
of the room. You have two thermometers at your disposal. Unfortunately, these thermometers deteriorate with age:
one is a year old and the other is five years old. As a result, the newer one provides quite accurate results whereas 
the older one provides less accurate results. 

Let's calculate the temperature of the room given these two measurements.


### Defining the problem as a graph

How can we define this problem as a graph? 

Well, the temperature of the room is hidden from you. You have only the readings
of the two thermometers at your disposal. Thinking about how thermometers work, we know that
their readings are influenced by two things; the temperature of the room and their inaccuracy
that we mentioned earlier.

```
     (Inaccuracy) ---- + -> (First Thermometer)  -> (Observation)
                      /
(Room Temperature) ---      
                      \
     (Inaccuracy) ---- + -> (Second Thermometer) -> (Observation) 
```

### Implementation

Let's assume we're on a distant planet and have no prior knowledge about what the temperature of the room may be, so 
we define the temperature as a Uniform Distribution between 20° and 30°.

```java
{% snippet TempExpectedValues %}
```

Let's now define our two thermometers. Looking at the graph we made earlier, we can see that each thermometer
is comprised of the room temperature and its given inaccuracy. 

Let's represent each thermometer as a Gaussian distribution with a mu of the room temperature and a sigma of its inaccuracy.
As the first thermometer is more accurate, its sigma will be smaller.

```java
{% snippet TempThermModel %}
```

Now we can take the readings of each thermometer.

```java
{% snippet TempObservations %}
```

Now that we have taken our thermometer readings, let's calculate the most probable value for the 
room temperature.

```java
{% snippet TempMostProbable %}
```

Given our graph and observed values, this will attempt to calculate the most likely value for each vertex.

### Results

Can you have a guess at what you think the result will be before you run the code?

The accurate thermometer is reporting 25° and the less accurate thermometer is reporting 30°. The 
temperature of the room is likely to be somewhere in the middle but weighted more to the accurate thermometer.

Running this code gives a value of:

```
26.01°
```

### Code

Here is the completed code if you'd like to run it yourself.

Experiment with the size of the sigma in each thermometer (the inaccuracy) and see how it affects the 
estimated temperature.

#### Java

```java
{% snippet TempFull %}
```

#### Python

```python
{% snippet PythonTwoThermometers %}
```
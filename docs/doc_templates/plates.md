---
# Page settings
layout: default
keywords: sequence template
comments: false
permalink: /docs/sequences/

# Hero section
title: Sequences
description: Using template notation can be a powerful way to describe your model

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/particle-filter/'
    next:
        content: Next page
        url: '/docs/autocorrelation/'

---

## What is a sequence?

A sequence is a group of vertices that is repeated multiple times in the network. The vertices in one group can 
(optionally) depend on vertices in the previous group. They typically represent a concept larger than a vertex like 
an agent in an ABM, a time series or some observations that are associated.

## How do you build them?

Let's say you have a class `MyData` that looks like this:
```java
{% snippet SequenceData %}
```
This is an example of how you could pull in data from a csv file and run linear regression, using
a sequence to build identical sections of the graph for each line of the csv.

```java
{% snippet Sequence %}
```
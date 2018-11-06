---
# Page settings
layout: default
keywords: plates
comments: false
permalink: /docs/plates/

# Hero section
title: Plates
description: Using plate notation can be a powerful way to describe your model

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/particle-filter/'
    next:
        content: Next page
        url: '/docs/examples/thermometer/'

---

## What is a plate?

A plate is a group of vertices that is repeated multiple times in the network. They typically
represent a concept larger than a vertex like an agent in an ABM, a time series or some observations that are
associated.

## How do you build them?

We're redesigning how this is done but for now there are some handy helper functions to get you
started.

Let's say you have a class `MyData` that looks like this:
```java
{% snippet PlatesData %}
```
This is an example of how you could pull in data from a csv file and run linear regression, using
plates to build identical sections of the graph for each line of the csv.

```java
{% snippet Plates %}
```
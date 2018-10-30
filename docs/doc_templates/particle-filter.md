---
# Page settings
layout: default
keywords: particle filter
comments: false
permalink: /docs/particle-filter/

# Hero section
title: Using a Particle Filter
description: Particle filters can help you find probable states of your network

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/regression/'
    next: 
        content: Next page
        url: '/docs/plates'
---

## Overview
Particle filters can help you find probable states of your network which can be used as starting states for MCMC sampling. 
Here is an example on how to use Keanu's Particle filter class:

```java
{% snippet ParticleFilterExample %}
```
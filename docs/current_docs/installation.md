---
# Page settings
layout: default
keywords: probabilistic getting started keanu
comments: false
permalink: /docs/installation/

# Hero section
title: Installation
description: Learn how to get Keanu installed in your project.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: 
        url: '#'
    next:
        content: Next page
        url: '/docs/overview/'

---

## Building a new model with Keanu

You should use the [Keanu starter project](https://github.com/improbable-research/keanu-starter)
 as a starting point for building new models on Keanu. The starter project includes the recommended file layout as 
 well as a properly configured build script using [Gradle](https://gradle.org/)

To create a new project from the starter project simply run the following command:
```
git clone --depth 1 https://github.com/improbable-research/keanu-starter.git
```

This clones the Keanu starter repo.

Now that you have the starter project, head over to [getting started]({{ site.baseurl }}/docs/getting-started) to get started.

## Using Keanu in an existing model (JVM based)

Keanu can be used with any build system that can pull artifacts from Maven Central (i.e. gradle, maven).

If you would like to start using Keanu in an existing project, simply add Keanu as a dependency 
in your gradle or maven build file.

#### Gradle

In your project's `build.gradle`

```
compile group: 'io.improbable', name: 'keanu', version: '0.0.15'
```

#### Maven

In your project's `pom.xml`

```
<dependency>
    <groupId>io.improbable</groupId>
    <artifactId>keanu</artifactId>
    <version>0.0.15</version>
</dependency>
```

Now that you have the dependency, head over to [getting started]({{ site.baseurl }}/docs/getting-started) to get started.

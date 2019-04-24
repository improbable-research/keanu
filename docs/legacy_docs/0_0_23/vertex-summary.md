---
# Page settings
layout: default
keywords: vertices
comments: false
version: 0.0.23
permalink: /docs/0_0_23/vertex-summary/

# Hero section
title: Vertices
description: The building blocks of Keanu are vertices. They are used to describe random variables and 
             the deterministic operations on them.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_23/data-io/'
    next:
        content: Next page
        url: '/docs/0_0_23/tensors/'
---

## Vertices

### Probabilistic

Probabilistic vertices are vertices that are not a deterministic function of their parents. An example
of this is a `GaussianVertex` which is non-deterministic and has a probability distribution controlled by 
parameters that come from its parents.

Changing the value of their parent vertices may change the probability density function but it 
will not directly cause a change of the value of the vertex.

### Non-Probabilistic

The value of these vertices are completely dependent on their parent vertices' values. For example,
given `C = A * B` (for any vertices A and B), C is a non-probabilistic vertex. Even if A or B are probabilistic 
vertices, C is still completely dependent on their values which means it is non-probabilistic.

## Vertex Families

There are several families of vertices. Each family shares a common value type.

### The Double Family

A `DoubleVertex` is one whose values are double-precision floating point numbers. It can be used by most arithmetic operators. 
They can be used to describe a problem that can be solved using gradient ascent optimization.

The currently available double vertices are:
- [Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/dbl/probabilistic/package-summary.html)
- [Non-Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/dbl/nonprobabilistic/package-summary.html)

### The Integer Family

An `IntegerVertex` is similar to the DoubleVertex except its value is an integer.

The currently available integer vertices are:
- [Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/intgr/probabilistic/package-summary.html)
- [Non-Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/intgr/nonprobabilistic/package-summary.html)

### The Boolean (true/false) Family

A `BooleanVertex` can be used by most boolean operators. These can be observed directly and used in [MCMC]({{ site.baseurl }}/docs/0_0_23/inference-posterior-sampling/).

The currently available boolean vertices are:
- [Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/bool/probabilistic/package-summary.html)
- [Non-Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/bool/nonprobabilistic/package-summary.html)

### The Generic (everything else) family

These are the vertices that can have any type as a value. For example, this type can be an Enum or any user defined object.
Let's look at an example of this in Keanu with the `CategoricalVertex` which will return a value of the specified Enum `MyType`.

```java
public enum MyType {
    A, B, C, D
}

public CategoricalVertex<MyType, GenericTensor<MyType>> getSelectorForMyType() {

    LinkedHashMap<MyType, DoubleVertex> frequency = new LinkedHashMap<>();
    frequency.put(A, new ConstantDoubleVertex(0.25));
    frequency.put(B, new ConstantDoubleVertex(0.25));
    frequency.put(C, new ConstantDoubleVertex(0.25));
    frequency.put(D, new ConstantDoubleVertex(0.25));

    return new CategoricalVertex<>(frequency);
}
```

The getSelectorForMyType() method returns a probabilistic vertex that would contain an 
object of type MyType A, B, C or D, 25% of the time respectively.

The currently available generic vertices are
- [Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/generic/probabilistic/package-summary.html)
- [Non-Probabilistic](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/vertices/generic/nonprobabilistic/package-frame.html)


### Tensors

Vertices also have a `shape`, which describes the tensor shape contained within them. A vertex with shape
[2,2] represents a 2 by 2 matrix. A vertex of shape [1,3] represents a row vector of length 3. The shape
can have any number of dimensions and any length.

Read more about tensors [here]({{ site.baseurl }}/docs/0_0_23/tensors) 

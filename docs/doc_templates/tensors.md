---
# Page settings
layout: default
keywords: tensors
comments: false
permalink: /docs/tensors/

# Hero section
title: Tensors (N Dimensional arrays)
description: Vertices can contain more than a single value. They can represent vectors,
             matrices or higher dimensional collections of data.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/vertex-summary/'
    next:
        content: Next page
        url: '/docs/inference-map/'
---

## Tensor rank and shape

Tensors in most cases can be thought of as nested arrays of values that can have any number
of dimensions. A tensor with one dimension can be thought of as a vector, a tensor
with two dimensions as a matrix and a tensor with three dimensions can be thought of as a cuboid. 
The number of dimensions a tensor has is called its `rank` and the length in each dimension 
describes its `shape`. 

For example, a 2 by 3 matrix:

```
1 2 3
4 5 6
```

has a `rank` of `2`, a `shape` of `[2, 3]` and a `length` of 6.

If you are struggling to get your head around the notion of a tensor, [this article](https://www.kdnuggets.com/2018/05/wtf-tensor.html) attempts to provide some intuition on what tensors are and how you can use them.

## Tensors in Keanu

Tensors can be extremely powerful as a way to represent large data sets or a way to very efficiently do the same
operation on many different pieces of data. This is because tensor operations can be done on the GPU.

For example, if we have two lists of numbers and some observation on their product then it's much more efficient
and much cleaner to describe this using tensors.

### Creating Tensors

Nearly everything in Keanu supports Tensors. But how do you create one?

Let's create vectors of doubles, integers and booleans that share the same value.

```java
{% snippet TensorSharedValue %}
```


Let's make some 2x2 matrices of doubles, integers and booleans.

```java
{% snippet Tensor2by2 %}
```

Want to change the shape of your tensor on the fly? You have to make sure the proposed shape is the same 
length as the original. For example, you can change a 2x2 tensor to a 1x4 tensor, but you can't change a 2x2 tensor
to a 2x3 tensor.

Here's how to do that in Keanu:

```java
{% snippet TensorReshape %}
```

### Tensor Operations

What operations can I apply to tensors?

Here's a small example of the power of tensors. All of these operations apply to each value in the tensor.

```java
{% snippet TensorOps %}
```

A complete list of tensor operations is available here:
- [Double Tensor](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/tensor/dbl/DoubleTensor.html)
- [Integer Tensor](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/tensor/intgr/IntegerTensor.html)
- [Boolean Tensor](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/tensor/bool/SimpleBooleanTensor.html)
- [Generic Tensor](https://static.javadoc.io/io.improbable/keanu/{{ site.current_version }}/io/improbable/keanu/tensor/generic/GenericTensor.html)


### Creating Vertices with Tensors

Let's say I want to create a vector of 100 Gaussians all with a mu of 0 and a sigma of 1.
This is how you do that in Keanu:

```java
{% snippet TensorVertexCreate %}
```

In this example, Keanu has expanded out its mu and sigma into a vector of 100 identical mus and sigmas.
It has then used this to create a GaussianVertex with a vector of 100 values.
Each of these values map to a Gaussian distribution with the corresponding mu and sigma from the vector of 100 identical mus and sigmas.

```java
{% snippet TensorVertexInspection %}
```

But what if you want each of those Gaussian distributions to have different mus and sigmas?
It is possible to instantiate a Gaussian vertex and rather than passing a scalar value for either mu or sigma, 
you can instead pass it a vector of mus and sigmas. 
In this case, you must make sure that the size of the mu and sigma vectors match the size of the GaussianVertex you are creating or are scalar.

```java
{% snippet TensorVector %}
```  


## Example of Tensors
Tensors can provide us with a more succinct way of describing problems and can allow us to solve problems computationally efficiently on the GPU. 
Without using tensors we have to iterate over the data and aggregate some results. 
```
a = unknown number around 0.5
b = unknown number around 1.5

c = 3
d = 4

observe that a * c = 6
observe that b * d = 12
```

or with tensors we could describe this as

```
a = [unknown ~0.5, unknown ~1.5]
b = [3, 4]
observe a * b = [6, 12]
```

In Keanu the above problem would be written as:

```java
{% snippet TensorFinal %}
```

## Python

We do not expose Tensors in the Python API. Numpy provides the same concept through its `ndarray` class. 

Most of the operations that used Tensor within Java can be replaced with `ndarray` in Python.
For example, to create a vertex:

```python
{% snippet PythonVertexFromNDArray %}
```

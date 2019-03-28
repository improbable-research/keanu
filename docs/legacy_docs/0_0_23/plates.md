---
# Page settings
layout: default
keywords: sequence template
comments: false
version: 0.0.23
permalink: /docs/0_0_23/sequences/

# Hero section
title: Sequences
description: Using template notation can be a powerful way to describe your model

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_23/particle-filter/'
    next:
        content: Next page
        url: '/docs/0_0_23/autocorrelation/'

---

## What is a sequence?

A sequence is a group of vertices that is repeated multiple times in the network. The vertices in one group can 
(optionally) depend on vertices in the previous group. They typically represent a concept larger than a vertex like 
an agent in an ABM, a time series or some observations that are associated.

# Examples

Here are some examples that will walk you through the process of developing with Sequences.

## Writing time series

This example shows you how you can write a simple time series model using Sequences

```java
DoubleVertex two = new ConstantDoubleVertex(2);

// Define the labels of vertices we will use in our Sequence
VertexLabel x1Label = new VertexLabel("x1");
VertexLabel x2Label = new VertexLabel("x2");

// Define labels for the Proxy Vertices which stand in for a Vertex from the previous SequenceItem.
// They will be automatically wired up when you construct the Sequence.
// i.e. these are the 'inputs' to our SequenceItem
VertexLabel x1InputLabel = SequenceBuilder.proxyFor(x1Label);
VertexLabel x2InputLabel = SequenceBuilder.proxyFor(x2Label);

// Define a factory method that creates proxy vertices using the proxy vertex labels and then uses these
// to define the computation graph of the Sequence.
// Note we have labeled the output vertices of this SequenceItem
Consumer<SequenceItem> factory = sequenceItem -> {
    DoubleProxyVertex x1Input = new DoubleProxyVertex(x1InputLabel);
    DoubleProxyVertex x2Input = new DoubleProxyVertex(x2InputLabel);

    DoubleVertex x1Output = x1Input.multiply(two).setLabel(x1Label);
    DoubleVertex x2Output = x2Input.plus(x1Output).setLabel(x2Label);

    sequenceItem.addAll(x1Input, x2Input, x1Output, x2Output);
};

// Create the starting values of our sequence
DoubleVertex x1Start = new ConstantDoubleVertex(4).setLabel(x1Label);
DoubleVertex x2Start = new ConstantDoubleVertex(4).setLabel(x2Label);
VertexDictionary dictionary = SimpleVertexDictionary.of(x1Start, x2Start);

Sequence sequence = new SequenceBuilder<Integer>()
    .withInitialState(dictionary)
    .count(5)
    .withFactory(factory)
    .build();

```

Note: by using the `.withFactories` method on the builder, rather than the `.withFactory`, it is possible
to have factories which use proxy input vertices which are defined in other factories.
i.e. your vertices can cross factories.

## Observing many associated data points

This example shows you how you can repeat logic over many observed data points which are associated by having a 
dependency on a common global value. 
Here they are the intercept and weights of a linear regression model. 


Let's say you have a class `MyData` that looks like this:
```java
public static class MyData {
    public double x;
    public double y;

    public MyData(String x, String y) {
        this.x = Double.parseDouble(x);
        this.y = Double.parseDouble(y);
    }
}
```
This is an example of how you could pull in data from a csv file and run linear regression, using
a sequence to build identical sections of the graph for each line of the csv.

```java

/**
 * Each sequence item contains a linear regression model:
 * VertexY = VertexX * VertexM + VertexB
 *
 * @param dataFileName The input data file defines, for each sequence item:
 *                     - the value of the input, VertexX
 *                     - the value of the observed output, VertexY
 */
public Sequence buildSequence(String dataFileName) {
    //Parse the csv data to MyData objects
    List<MyData> allMyData = ReadCsv.fromFile(dataFileName)
        .asRowsDefinedBy(MyData.class)
        .load();

    DoubleVertex m = new GaussianVertex(0, 1);
    DoubleVertex b = new GaussianVertex(0, 1);
    VertexLabel xLabel = new VertexLabel("x");
    VertexLabel yLabel = new VertexLabel("y");

    //Build sequence from each line in the csv
    Sequence sequence = new SequenceBuilder<MyData>()
        .fromIterator(allMyData.iterator())
        .withFactory((item, csvMyData) -> {

            ConstantDoubleVertex x = new ConstantDoubleVertex(csvMyData.x).setLabel(xLabel);
            DoubleVertex y = m.multiply(x).plus(b).setLabel(yLabel);

            DoubleVertex yObserved = new GaussianVertex(y, 1);
            yObserved.observe(csvMyData.y);

            // this labels the x and y vertex for later use
            item.add(x);
            item.add(y);
        })
        .build();

    //now you have access to the "x" from any one of the sequence
    DoubleTensor valueForXAtCSVLine1 = sequence.asList()
        .get(1) // get sequence item 1 which is built from csv line 1
        .<DoubleVertex>get(xLabel) //get the vertex that we labelled "x" in that item
        .getValue(); //get the value from that vertex

    //Now run an inference algorithm on vertex m and vertex b and you have linear regression

    return sequence;
}
```

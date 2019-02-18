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
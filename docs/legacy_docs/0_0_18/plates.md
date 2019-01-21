---
# Page settings
layout: default
keywords: plates
comments: false
version: 0.0.18
permalink: /docs/0_0_18/plates/

# Hero section
title: Plates
description: Using plate notation can be a powerful way to describe your model

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/0_0_18/particle-filter/'
    next:
        content: Next page
        url: '/docs/0_0_18/autocorrelation/'

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
plates to build identical sections of the graph for each line of the csv.

```java

/**
 * Each plate contains a linear regression model:
 * VertexY = VertexX * VertexM + VertexB
 *
 * @param dataFileName The input data file defines, for each plate:
 *                     - the value of the input, VertexX
 *                     - the value of the observed output, VertexY
 */
public Plates buildPlates(String dataFileName) {
    //Parse the csv data to MyData objects
    List<MyData> allMyData = ReadCsv.fromFile(dataFileName)
        .asRowsDefinedBy(MyData.class)
        .load();

    DoubleVertex m = new GaussianVertex(0, 1);
    DoubleVertex b = new GaussianVertex(0, 1);
    VertexLabel xLabel = new VertexLabel("x");
    VertexLabel yLabel = new VertexLabel("y");

    //Build plates from each line in the csv
    Plates plates = new PlateBuilder<MyData>()
        .fromIterator(allMyData.iterator())
        .withFactory((plate, csvMyData) -> {

            ConstantDoubleVertex x = new ConstantDoubleVertex(csvMyData.x).setLabel(xLabel);
            DoubleVertex y = m.multiply(x).plus(b).setLabel(yLabel);

            DoubleVertex yObserved = new GaussianVertex(y, 1);
            yObserved.observe(csvMyData.y);

            // this labels the x and y vertex for later use
            plate.add(x);
            plate.add(y);
        })
        .build();

    //now you have access to the "x" from any one of the plates
    DoubleTensor valueForXAtCSVLine1 = plates.asList()
        .get(1) // get plate 1 which is built from csv line 1
        .<DoubleVertex>get(xLabel) //get the vertex that we labelled "x" in that plate
        .getValue(); //get the value from that vertex

    //Now run an inference algorithm on vertex m and vertex b and you have linear regression

    return plates;
}
```
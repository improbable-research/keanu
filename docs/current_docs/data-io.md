---
# Page settings
layout: default
keywords: data
comments: false
permalink: /docs/data-io/

# Hero section
title: "Reading/Writing CSV"
description: Your model should be data-centric. Keanu can help you with that.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/getting-started/'
    next:
        content: Next page
        url: '/docs/vertex-summary/'

---


## Reading data into and out of Keanu

This page tells you how to:
- Load a CSV data-set into your model
- Save vertex values to CSV
- Save samples to CSV

### Reading CSV data

Keanu is packaged with a CSV reader that allows you to load simple CSV files with or without
a header. You can read from a specified file, predefined string or a file on the class path.

We will go through the steps of how to read from a file using a CsvReader and then how to
extract and parse the lines from the reader.

#### Creating a CsvReader from your CSV data
To read from a file from your home directory:

```java
CsvReader reader1 = ReadCsv.fromFile("~/my_filename.csv");
```

If you place the CSV file on your classpath (e.g. in src/main/resources), you can load it as a resource,
which allows you to avoid having to provide a hardcoded file path.

```java
CsvReader reader2 = ReadCsv.fromResources("my_other_filename.csv");
```

Once you have a `CsvReader`, you can call `readLines()` to start reading each line.
```java
for (List<String> csvLine : reader1.readLines()) {
    //do something with your csv line
}
```

#### Large CSV files

If your CSV file is very large, you may not want to load the entirety of it into memory before
processing it. You can stream the lines in order to avoid holding the entire file in memory.

Once you have a `CsvReader`, you can call `streamLines()` to start streaming each line. Make sure
to close the stream as it is potentially connected to an open file. Closing the stream can be
done by using *try-with-resources*, as shown below.

```java
try (Stream<List<String>> lineStream = reader2.streamLines()) {

    lineStream.forEach(line -> {
        //do something with your line.
    });
}
```

### Reading CSV as a plain old java object (POJO)

If you don't feel like processing the raw lines from your CSV file, then you 
have the option to read directly to a Java object.

Given a POJO

```java
public class MyClass {
    public String myString;
    public int myInt;
}
```

And some CSV

```
myString,myInt
aString,1
bString,2
cString,3
```

The CSV can be loaded as a Java object by

```java
List<MyClass> myPojos = ReadCsv.fromFile("some/file/path")
    .asRowsDefinedBy(MyClass.class)
    .load();
```

If your CSV header names contain illegal characters you have the option to
tag the field with a Java annotation.

Given CSV

```
my-problematic*header,myInt
aString,1
bString,2
cString,3
```

Annotate your class with `@CsvProperty` and the problematic header title.

```java
public class MyOtherClass {
    @CsvProperty("my-problematic*header")
    public String myString;
    public int myInt;
}
```

The CSV parser also supports using setter methods.

```java
public class MySetterClass {
    private String myString;

    public void setMyString(String aString) {
        myString = aString;
    }
}
```


### Reading into a model

Take a look at [Diabetes Linear Regression](https://github.com/improbable-research/keanu/blob/develop/keanu-project/src/test/java/io/improbable/keanu/e2e/regression/DiabetesLinearRegression.java) 
for a complete example of reading CSV data into a model in Keanu.

## Writing out data

Keanu is packaged with a CSV writer that allows you to write to a CSV file. This is useful for writing values stored in vertices
or samples to a file.

### Creating a CsvWriter from vertices

This section assumes knowledge of tensors - please look [here](https://nd4j.org/userguide#intro) to familiarise yourself with them first. 
Only scalar and vector tensors are supported by the CSV writer at the moment.

Before looking at the different ways in which we can write values stored in vertices to CSV files, 
we have to decide which output format we want to use. The options are:

* Column format: each vertex occupies a column in the CSV file, with each row denoting the index of the vector.
* Row format:  each vertex occupies a row, with each column denoting the index of the vector.

Let's look at an example of each format.
Say, we have run inference on a model and would like to write the resulting (most probable) values to a CSV file and 
that the list of inferred variables contains the following three vertices:

```java
List<DoubleVertex> inferredVariables = runMyModel();

inferredVariables.get(0); // [0.5, 1.0, 1.5, 2.0]
inferredVariables.get(1); // [5.0, 10.0, 15.0, 20.0]
inferredVariables.get(2); // [50.0, 100.0, 150.0]
```

Let's write these as columns to a file called `columnTest.csv` in the root directory.

```java
File file = WriteCsv.asColumns(inferredVariables).toFile("columnTest.csv");
```

The CSV file `columnTest.csv` contains:

```text
0.5,5.0,50.0
1.0,10.0,100.0
1.5,15.0,150.0
2.0,20.0,-
```

Let's now write these as rows to a file called `rowTest.csv` in the root directory.

```java
File file2 = WriteCsv.asRows(inferredVariables).toFile("rowTest.csv");
```

The CSV file rowTest.csv contains:

```text
0.5,1.0,1.5,2.0
5.0,10.0,15.0,20.0
50.0,100.0,150.0,-
```

### Default headers

It's quite difficult trying to decipher and remember what the columns contain without a header. 
Fortunately, the CSV writer provides default headers that provide information about each column.
Let's re-run the above examples but using the default headers provided in the CSV writer.

As columns.

```java
File file3 = WriteCsv.asColumns(inferredVariables).withDefaultHeader().toFile("columnTest.csv");
```

The default header sets the column name to the vertex ID.

```text
20, 21, 22
0.5,5.0,50.0
1.0,10.0,100.0
1.5,15.0,150.0
2.0,20.0,-
```

As rows.

```java
File file4 = WriteCsv.asRows(inferredVariables).withDefaultHeader().toFile("rowTest.csv");
```

The default header sets the column name to the index.

```text
[0], [1], [2], [3]
0.5,1.0,1.5,2.0
5.0,10.0,15.0,20.0
50.0,100.0,150.0,-
```

### Creating a CsvWriter from samples

Let's look at how you can write samples to CSV. 
First, you'll need some samples - let's take some from a network using the NUTS algorithm. 
To learn more about the parameters being used here, head over to the [NUTS documentation]({{ site.baseurl }}/docs/inference-posterior-sampling#nuts).

```java
ProbabilisticModel model = new KeanuProbabilisticModel(aVertex.getConnectedGraph());
NetworkSamples samples = Keanu.Sampling.NUTS.withDefaultConfig().getPosteriorSamples(
    model,
    aVertex,
    1000
);
```

Now, we want to save these for analysis later on. How do we write these to CSV?
We recommend you write samples to file with the default CSV header.

```java
File file5 = WriteCsv.asSamples(samples, inferredVariables).withDefaultHeader().toFile("test.csv");
```

These are written to CSV in the format of `{vertex ID}[{index}]`, with each row representing a sample.

```java
2[0], 2[1], 2[2], 2[3], 2[4], 5[0], 5[1], 5[2], 5[3]
1.0,2.0,3.0,4.0,5.0,5.0,4.0,3.0,2.0
2.0,4.0,6.0,8.0,10.0,10.0,8.0,6.0,4.0
2.5,5.0,7.5,10.0,12.5,12.5,10.0,7.5,5.0
3.0,6.0,9.0,12.0,15.0,15.0,12.0,9.0,6.0
```




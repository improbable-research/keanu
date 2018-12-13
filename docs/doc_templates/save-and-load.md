---
# Page settings
layout: default
keywords: save load model
comments: false
permalink: /docs/save-and-load/

# Hero section
title: Saving and Loading Models
description: How to save models out and load them back in.

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/docs/autocorrelation/'
    next:
        content: Next page
        url: '/docs/examples/thermometer/'

---

## Saving and Loading models

Keanu supports the ability to save models to JSON, Dot and Protobuf formats for long term storage or transmission across
a network.  It also supports instantiating a network from a saved Protobuf & JSON format model.

Models can also be saved with their associated data and observations, allowing a user to save a pre-trained/optimised
model ready for future analysis.

Protobuf is the most efficient format, but is not human readable.  It is recommended to use this format for storage and
transmission purposes.

JSON should be used where a user wants to be able to manually view/search/edit the stored model.

Dot is useful when a user wishes to use existing graph visualisation tooling to inspect the underlying graph of the
model.

### Saving Examples

To save a model, a user simply has to create a NetworkSaver object and call the .save() method, passing in an OutputStream
and indicating whether they wish to save the current state of the model or to strip out all value information.  For
example to save as a Protobuf:
```java
{% snippet SaveToProtobuf %}
```

For JSON:
```java
{% snippet SaveToJSON %}
```

And similarly for Dot:
```java
{% snippet SaveToDot %}
```

The Dot saver also has the option to only output Vertices within a certain distance of a starting Vertex (eg if
a user requested a distance of 1 then the Dot output would contain the Vertex itself and its direct parents and children).
This operation is similarly simple:
```java
{% snippet SavePartialToDot %}
```

### Loading Examples

Loading a network is once again simple with a user creating an instance of a NetworkLoader object and calling the .load()
method.  This method will create a new BayesianNetwork object containing all the specified vertices and will replay the values if present
in the stored model.  This can be achieved in a few lines of code as below:

```java
{% snippet LoadFromProtobuf %}
```

```java
{% snippet LoadFromJSON %}
```

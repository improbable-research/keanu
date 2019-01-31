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

Keanu supports the ability to save models to JSON, Dot and Protobuf formats for long term storage, transmission across
a network or for visualisation purposes.  It also supports instantiating a network from a saved Protobuf & JSON format model.

Models can also be saved with their associated data and observations, allowing a user to save a pre-trained/optimised
model ready for future analysis.

Protobuf is the most efficient format, but is not human readable.  It is recommended to use this format for storage and
transmission purposes.

JSON should be used where a user wants to be able to manually view/search/edit the stored model.

Dot is useful when a user wishes to use existing graph visualization tooling to inspect the underlying graph of the
model.

### Java Saving Examples

To save a model, a user simply has to create a NetworkSaver object and call the .save() method, passing in an OutputStream
and indicating whether they wish to save the current state of the model or to strip out all value information.  For
example to save as a Protobuf:
```java
public void saveNetToProtobuf(BayesianNetwork net,
                              OutputStream outputStream,
                              boolean saveValuesAndObservations) throws IOException {
    NetworkSaver saver = new ProtobufSaver(net);
    saver.save(outputStream, saveValuesAndObservations);
}
```

For JSON:
```java
public void saveNetToJSON(BayesianNetwork net,
                          OutputStream outputStream,
                          boolean saveValuesAndObservations) throws IOException {
    NetworkSaver saver = new JsonSaver(net);
    saver.save(outputStream, saveValuesAndObservations);
}
```

And similarly for Dot:
```java
public void saveNetToDotFile(BayesianNetwork net,
                             OutputStream outputStream,
                             boolean saveValuesAndObservations) throws IOException {
    NetworkSaver saver = new DotSaver(net);
    saver.save(outputStream, saveValuesAndObservations);
}
```

The Dot saver also has the option to only output Vertices within a certain distance of a starting Vertex (eg if
a user requested a distance of 1 then the Dot output would contain the Vertex itself and its direct parents and children).
This operation is similarly simple:
```java
public void savePartialNetToDot(Vertex startingVertex,
                                int degree,
                                BayesianNetwork net,
                                OutputStream outputStream,
                                boolean saveValuesAndObservations) throws IOException {
    DotSaver saver = new DotSaver(net);
    saver.save(outputStream, startingVertex, degree, saveValuesAndObservations);
}
```

### Java Loading Examples

Loading a network is once again simple with a user creating an instance of a NetworkLoader object and calling the .load()
method.  This method will create a new BayesianNetwork object containing all the specified vertices and will replay the values if present
in the stored model.  This can be achieved in a few lines of code as below:

```java
public BayesianNetwork loadNetFromProtobuf(InputStream input) throws IOException {
    NetworkLoader loader = new ProtobufLoader();
    return loader.loadNetwork(input);
}
```

```java
public BayesianNetwork loadNetFromJSON(InputStream input) throws IOException {
    NetworkLoader loader = new JsonLoader();
    return loader.loadNetwork(input);
}
```

### Python Saving Examples

Saving a model in Python is also straightforward requiring only that you construct your saver and pass it a fully
qualified file name and the network to save (along with an optional metadata map).  An example for each of the savers
is given below:

```python
net = BayesNet(gamma.get_connected_graph())
metadata = {"Author": "Documentation Team"}

protobuf_saver = ProtobufSaver(net)
protobuf_saver.save(PROTO_FILE_NAME, True, metadata)

json_saver = JsonSaver(net)
json_saver.save(JSON_FILE_NAME, True, metadata)

dot_saver = DotSaver(net)
dot_saver.save(DOT_FILE_NAME, True, metadata)
```

### Python Loading Examples

Once again, networks can be loaded using Python API via the JSON or Protobuf Loader objects.  Example below:

```python
protobuf_loader = ProtobufLoader()
new_net_from_proto = protobuf_loader.load(PROTO_FILE_NAME)
json_loader = JsonLoader()
new_net_from_json = json_loader.load(JSON_FILE_NAME)
```

package io.improbable.snippet;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.util.graph.VertexGraph;
import io.improbable.keanu.util.graph.io.GraphToDot;
import io.improbable.keanu.util.io.JsonLoader;
import io.improbable.keanu.util.io.JsonSaver;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.io.ProtobufSaver;
import io.improbable.keanu.vertices.Vertex;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class SaveLoadExamples {

    //%%SNIPPET_START%% SaveToProtobuf
    public void saveNetToProtobuf(BayesianNetwork net,
                                  OutputStream outputStream,
                                  boolean saveValuesAndObservations) throws IOException {
        NetworkSaver saver = new ProtobufSaver(net);
        saver.save(outputStream, saveValuesAndObservations);
    }
    //%%SNIPPET_END%% SaveToProtobuf

    //%%SNIPPET_START%% SaveToJSON
    public void saveNetToJSON(BayesianNetwork net,
                              OutputStream outputStream,
                              boolean saveValuesAndObservations) throws IOException {
        NetworkSaver saver = new JsonSaver(net);
        saver.save(outputStream, saveValuesAndObservations);
    }
    //%%SNIPPET_END%% SaveToJSON

    //%%SNIPPET_START%% SaveToDot
    public void saveNetToDotFile(BayesianNetwork net,
                                 OutputStream outputStream) throws IOException {
        VertexGraph graph = new VertexGraph(net);
        GraphToDot.write(graph, outputStream);
    }
    //%%SNIPPET_END%% SaveToDot

    //%%SNIPPET_START%% SavePartialToDot
    public void savePartialNetToDot(Vertex startingVertex,
                                    int degree,
                                    BayesianNetwork net,
                                    OutputStream outputStream,
                                    boolean saveValuesAndObservations) throws IOException {
        DotSaver saver = new DotSaver(net);
        saver.save(outputStream, startingVertex, degree, saveValuesAndObservations);
    }
    //%%SNIPPET_END%% SavePartialToDot

    //%%SNIPPET_START%% LoadFromProtobuf
    public BayesianNetwork loadNetFromProtobuf(InputStream input) throws IOException {
        NetworkLoader loader = new ProtobufLoader();
        return loader.loadNetwork(input);
    }
    //%%SNIPPET_END%% LoadFromProtobuf

    //%%SNIPPET_START%% LoadFromJSON
    public BayesianNetwork loadNetFromJSON(InputStream input) throws IOException {
        NetworkLoader loader = new JsonLoader();
        return loader.loadNetwork(input);
    }
    //%%SNIPPET_END%% LoadFromJSON
}

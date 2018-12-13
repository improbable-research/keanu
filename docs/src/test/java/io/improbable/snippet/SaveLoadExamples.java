package io.improbable.snippet;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.util.io.DotSaver;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.io.ProtobufSaver;
import io.improbable.keanu.vertices.Vertex;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class SaveLoadExamples {

    public void saveModelToProtobuf(BayesianNetwork net,
                                    OutputStream outputStream,
                                    boolean saveValuesAndObservations) throws IOException {
        NetworkSaver saver = new ProtobufSaver(net);
        saver.save(outputStream, saveValuesAndObservations);
    }

    public void saveNetToDotFile(BayesianNetwork net,
                                 OutputStream outputStream,
                                 boolean saveValuesAndObservations) throws IOException {
        NetworkSaver saver = new DotSaver(net);
        saver.save(outputStream, saveValuesAndObservations);
    }

    public void savePartialNetworkToDot(Vertex startingVertex,
                                        int degree,
                                        BayesianNetwork net,
                                        OutputStream outputStream,
                                        boolean saveValuesAndObservations) throws IOException {
        DotSaver saver = new DotSaver(net);
        saver.save(outputStream, startingVertex, degree, saveValuesAndObservations);
    }

    public BayesianNetwork loadNetFromProtobuf(InputStream input) throws IOException {
        NetworkLoader loader = new ProtobufLoader();
        return loader.loadNetwork(input);
    }
}

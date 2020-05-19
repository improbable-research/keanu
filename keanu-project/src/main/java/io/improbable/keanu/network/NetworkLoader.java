package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.io.IOException;
import java.io.InputStream;

public interface NetworkLoader {

    BayesianNetwork loadNetwork(InputStream input) throws IOException;

    void loadValue(Vertex vertex);
}

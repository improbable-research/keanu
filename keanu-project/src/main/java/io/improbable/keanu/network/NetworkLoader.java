package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.InputStream;

public interface NetworkLoader {

    BayesianNetwork loadNetwork(InputStream input) throws IOException;

    void loadValue(DoubleVertex vertex);
    void loadValue(BooleanVertex vertex);
    void loadValue(IntegerVertex vertex);
    void loadValue(Vertex vertex);
}

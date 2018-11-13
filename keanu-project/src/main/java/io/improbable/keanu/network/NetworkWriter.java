package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.io.IOException;
import java.io.OutputStream;

public interface NetworkWriter {

    void save(OutputStream output, boolean saveValues) throws IOException;
    void save(Vertex vertex);

    default void save(ConstantDoubleVertex vertex) {
        save((ConstantVertex)vertex);
    }

    void save(ConstantVertex vertex);
    void saveValue(Vertex vertex);
    void saveValue(DoubleVertex vertex);

}

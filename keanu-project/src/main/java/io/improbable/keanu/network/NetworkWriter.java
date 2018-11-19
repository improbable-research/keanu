package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.OutputStream;

public interface NetworkWriter {

    void save(OutputStream output, boolean saveValues) throws IOException;
    void save(Vertex vertex);

    default void save(OutputStream output, Vertex vertex, int degree, boolean saveValues) throws IOException {
        save(output, saveValues);
    }

    void save(ConstantVertex vertex);
    default void save(ConstantDoubleVertex vertex) {
        save((ConstantVertex)vertex);
    }

    void saveValue(Vertex vertex);
    void saveValue(DoubleVertex vertex);
    void saveValue(IntegerVertex vertex);
    void saveValue(BoolVertex vertex);
}

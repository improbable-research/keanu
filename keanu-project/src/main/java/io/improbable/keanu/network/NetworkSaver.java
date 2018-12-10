package io.improbable.keanu.network;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;

public interface NetworkSaver {

    void save(OutputStream output, boolean saveValues, Map<String, String> metadata) throws IOException;
    default void save(OutputStream output, boolean saveValues) throws IOException {
        save(output, saveValues, null);
    }
    void save(Vertex vertex);

    default void save(ConstantVertex vertex) {
        save((Vertex)vertex);
    }

    default void save(ConstantDoubleVertex vertex) {
        save((ConstantVertex)vertex);
    }

    default void save(ConstantIntegerVertex vertex) {
        save((ConstantVertex)vertex);
    }

    default void save(ConstantBoolVertex vertex) {
        save((ConstantVertex)vertex);
    }

    void saveValue(Vertex vertex);

    default void saveValue(DoubleVertex vertex) {
        saveValue((Vertex)vertex);
    }

    default void saveValue(IntegerVertex vertex)  {
        saveValue((Vertex)vertex);
    }

    default void saveValue(BoolVertex vertex)  {
        saveValue((Vertex)vertex);
    }
}

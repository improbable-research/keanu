package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface NetworkReader {

    void loadValue(DoubleVertex vertex);
    void loadValue(BoolVertex vertex);
    void loadValue(IntegerVertex vertex);
    void loadValue(Vertex vertex);
}

package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;

import java.io.IOException;
import java.io.OutputStream;

public interface NetworkWriter {

    public void save(OutputStream output, boolean saveValues) throws IOException;
    public void save(Vertex vertex);
    public void save(ConstantDoubleVertex vertex);
    public void saveValue(Vertex vertex);
    public void saveValue(DoubleVertex vertex);

}

package io.improbable.keanu.backend;

import io.improbable.keanu.vertices.Vertex;

public interface GraphBuilder<T extends ComputableGraph> {

    void createConstant(Vertex visiting);

    void createVariable(Vertex visiting);

    void convert(Vertex visiting);

    T build();
}

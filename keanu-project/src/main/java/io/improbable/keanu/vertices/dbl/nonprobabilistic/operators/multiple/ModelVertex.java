package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import sun.security.provider.certpath.Vertex;

import java.util.Map;
import java.util.function.Function;

public interface ModelVertex<T> extends NonProbabilistic<T> {

    void run(Map<VertexLabel, DoubleVertex> inputs);

    Map<VertexLabel, Double> updateValues(Map<VertexLabel, DoubleVertex> inputs);

    Double getModelOutputValue(VertexLabel label);

    DoubleVertex getModelOutputVertex(VertexLabel label);

}

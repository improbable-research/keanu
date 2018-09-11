package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;

public interface ModelVertex<T> extends NonProbabilistic<T> {

    Map<VertexLabel, Double> run();

    Double getModelOutputValue(VertexLabel label);

    DoubleVertex getModelOutputVertex(VertexLabel label);

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;
import java.util.function.Function;

public interface ModelProcessVertex<T> extends ModelVertex<T> {

    String process(Map<VertexLabel, DoubleVertex> inputs);

    Map<VertexLabel, Double> run(String process, Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput);

}

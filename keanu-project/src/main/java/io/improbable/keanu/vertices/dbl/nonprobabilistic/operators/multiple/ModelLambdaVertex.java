package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;
import java.util.function.Function;

public interface ModelLambdaVertex extends ModelVertex {

    Map<VertexLabel, Double> run(Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> lambda);

}

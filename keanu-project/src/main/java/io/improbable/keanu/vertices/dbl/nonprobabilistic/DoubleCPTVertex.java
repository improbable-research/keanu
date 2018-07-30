package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.number.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTVertex;

import java.util.List;
import java.util.Map;

public class DoubleCPTVertex extends CPTVertex<DoubleTensor> {

    public DoubleCPTVertex(List<Vertex<? extends Tensor<Boolean>>> inputs,
                           Map<Condition, DoubleVertex> conditions,
                           DoubleVertex defaultResult) {
        super(inputs, conditions, defaultResult);
    }
}

package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;

public class AndMultipleVertex extends BoolReduceVertex {
    public AndMultipleVertex(Collection<Vertex<BooleanTensor>> input) {
        super(input, io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple.AndMultipleVertex::and);
    }

    private static BooleanTensor and(BooleanTensor a, BooleanTensor b) {
        return a.and(b);
    }
}

package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple.BoolReduceVertex;

import java.util.Collection;

public class OrMultipleVertex extends BoolReduceVertex {

    public OrMultipleVertex(Collection<Vertex<BooleanTensor>> input) {
        super(input, io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.multiple.OrMultipleVertex::or);
    }

    private static BooleanTensor or(BooleanTensor a, BooleanTensor b) {
        return a.or(b);
    }
}

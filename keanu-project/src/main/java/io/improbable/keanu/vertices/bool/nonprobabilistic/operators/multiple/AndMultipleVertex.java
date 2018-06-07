package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;

public class AndMultipleVertex extends BoolReduceVertex {
    public AndMultipleVertex(Collection<Vertex<BooleanTensor>> input) {
        super(checkAllShapesMatch(
            input.stream().map(Vertex::getShape).collect(Collectors.toList())
        ), input, AndMultipleVertex::and);
    }

    private static BooleanTensor and(BooleanTensor a, BooleanTensor b) {
        return a.and(b);
    }
}

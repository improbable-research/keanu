package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;

import java.util.Collection;
import java.util.stream.Collectors;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

public class OrMultipleVertex extends BooleanReduceVertex {

    public OrMultipleVertex(Collection<Vertex<BooleanTensor>> input) {
        super(checkAllShapesMatch(
            input.stream().map(Vertex::getShape).collect(Collectors.toList())
            ),
            input, (a,b) -> a.or(b));
    }
}

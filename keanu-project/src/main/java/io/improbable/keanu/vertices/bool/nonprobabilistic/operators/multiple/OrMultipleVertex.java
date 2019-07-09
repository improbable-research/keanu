package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;

import java.util.Collection;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;

public class OrMultipleVertex extends BooleanReduceVertex {

    public OrMultipleVertex(Collection<? extends IVertex<BooleanTensor>> input) {
        super(checkAllShapesMatch(
            input.stream().map(IVertex::getShape).collect(Collectors.toList())
            ),
            input, BooleanTensor::or);
    }
}

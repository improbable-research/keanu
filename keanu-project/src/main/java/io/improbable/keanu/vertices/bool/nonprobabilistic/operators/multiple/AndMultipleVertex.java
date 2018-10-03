package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import java.util.Collection;
import java.util.stream.Collectors;

public class AndMultipleVertex extends BoolReduceVertex {
    public AndMultipleVertex(Collection<Vertex<BooleanTensor>> input) {
        super(
                checkAllShapesMatch(
                        input.stream().map(Vertex::getShape).collect(Collectors.toList())),
                input,
                BooleanTensor::and);
    }
}

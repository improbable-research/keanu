package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.shapeAlongDimension;

public class BoolTADVertex extends BoolUnaryOpVertex<BooleanTensor> {

    private final int dimension;
    private final int index;

    /**
     * Takes the tensor along a dimension of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension the dimension to extract along
     * @param index the index of extraction
     */
    public BoolTADVertex(BoolVertex inputVertex, int dimension, int index) {
        super(shapeAlongDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected BooleanTensor op(BooleanTensor a) {
        return a.tad(dimension, index);
    }

}

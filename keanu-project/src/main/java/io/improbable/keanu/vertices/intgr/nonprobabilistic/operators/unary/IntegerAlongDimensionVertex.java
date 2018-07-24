package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.shapeAlongDimension;

public class IntegerAlongDimensionVertex extends IntegerUnaryOpVertex {

    private final int dimension;
    private final int index;

    /**
     * Takes the tensor along a dimension for a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension the dimension to extract along
     * @param index the index of extraction
     */
    public IntegerAlongDimensionVertex(IntegerVertex inputVertex, int dimension, int index) {
        super(shapeAlongDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected IntegerTensor op(IntegerTensor a) {
        return a.alongDimension(dimension, index);
    }

}

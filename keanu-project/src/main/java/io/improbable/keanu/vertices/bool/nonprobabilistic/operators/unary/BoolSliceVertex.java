package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

import static io.improbable.keanu.tensor.TensorShape.shapeSlice;

public class BoolSliceVertex extends BoolUnaryOpVertex<BooleanTensor> {

    private final int dimension;
    private final int index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension the dimension to extract along
     * @param index the index of extraction
     */
    public BoolSliceVertex(BoolVertex inputVertex, int dimension, int index) {
        super(shapeSlice(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected BooleanTensor op(BooleanTensor input) {
        return input.slice(dimension, index);
    }

}

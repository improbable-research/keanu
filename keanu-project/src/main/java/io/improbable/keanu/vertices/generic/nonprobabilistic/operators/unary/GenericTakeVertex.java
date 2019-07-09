package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;

public class GenericTakeVertex<T> extends GenericTensorUnaryOpVertex<T, T> {

    private static final String INDEX_NAME = "index";
    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex
     * @param index       the index of extraction
     */
    public GenericTakeVertex(@LoadVertexParam(INPUT_NAME) IVertex<GenericTensor<T>> inputVertex,
                             @LoadVertexParam(INDEX_NAME) long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
        this.index = index;
    }

    protected GenericTensor<T> op(GenericTensor<T> input) {
        return input.take(index);
    }

    @SaveVertexParam(INDEX_NAME)
    public long[] getIndex() {
        return index;
    }
}

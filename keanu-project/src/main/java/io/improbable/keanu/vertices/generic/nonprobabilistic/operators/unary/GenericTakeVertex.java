package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;

public class GenericTakeVertex<T, TENSOR extends Tensor<T, TENSOR>> extends GenericTensorUnaryOpVertex<T, TENSOR, T, TENSOR> {

    private static final String INDEX_NAME = "index";
    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex
     * @param index       the index of extraction
     */
    public GenericTakeVertex(@LoadVertexParam(INPUT_NAME) Vertex<TENSOR> inputVertex,
                             @LoadVertexParam(INDEX_NAME) long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
        this.index = index;
    }

    protected TENSOR op(TENSOR input) {
        return input.take(index);
    }

    @SaveVertexParam(INDEX_NAME)
    public long[] getIndex() {
        return index;
    }
}

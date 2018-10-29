package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class GenericTakeVertex<T> extends UnaryOpVertex<Tensor<T>,Tensor<T>> {

    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex
     * @param index       the index of extraction
     */
    public GenericTakeVertex(Vertex<Tensor<T>> inputVertex, long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
        this.index = index;
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected Tensor<T> op(Tensor<T> input) {
        return GenericTensor.scalar(input.getValue(index));
    }

}

package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShape.shapeSlice;

public class GenericSliceVertex<T> extends UnaryOpVertex<Tensor<T>, Tensor<T>> {

    private final int dimension;
    private final int index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public GenericSliceVertex(Vertex<Tensor<T>> inputVertex, int dimension, int index) {
        super(shapeSlice(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected Tensor<T> op(Tensor<T> input) {
        return input.slice(dimension, index);
    }

}

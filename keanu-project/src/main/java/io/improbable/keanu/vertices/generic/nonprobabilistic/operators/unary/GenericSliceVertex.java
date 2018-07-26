package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

import static io.improbable.keanu.tensor.TensorShape.shapeAlongDimension;

public class GenericSliceVertex<T> extends NonProbabilistic<Tensor<T>> {

    private final Vertex<? extends Tensor<T>> inputVertex;
    private final int dimension;
    private final int index;

    /**
     * Takes the tensor along a dimension of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public GenericSliceVertex(Vertex<? extends Tensor<T>> inputVertex, int dimension, int index) {

        this.inputVertex = inputVertex;
        this.dimension = dimension;
        this.index = index;
        setParents(inputVertex);
        setValue(Tensor.placeHolder(shapeAlongDimension(dimension, inputVertex.getShape())));
    }

    @Override
    public Tensor<T> getDerivedValue() {
        return op(inputVertex.getValue());
    }

    @Override
    public Tensor<T> sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected Tensor<T> op(Tensor<T> input) {
        return input.slice(dimension, index);
    }

}

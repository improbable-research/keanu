package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class GenericSliceVertex<T> extends UnaryOpVertex<Tensor<T>, Tensor<T>> {

    private static final String DIMENSION_NAME = "dimension";
    private static final String INDEX_NAME = "index";

    private final int dimension;
    private final int index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public GenericSliceVertex(@LoadVertexParam(INPUT_NAME) Vertex<Tensor<T>> inputVertex,
                              @LoadVertexParam(DIMENSION_NAME) int dimension,
                              @LoadVertexParam(INDEX_NAME) int index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex);
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

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return dimension;
    }

    @SaveVertexParam(INDEX_NAME)
    public int getIndex() {
        return index;
    }
}

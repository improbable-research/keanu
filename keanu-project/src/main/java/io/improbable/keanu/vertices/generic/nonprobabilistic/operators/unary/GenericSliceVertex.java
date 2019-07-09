package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class GenericSliceVertex<T> extends GenericTensorUnaryOpVertex<T, T> {

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
    public GenericSliceVertex(@LoadVertexParam(INPUT_NAME) IVertex<GenericTensor<T>> inputVertex,
                              @LoadVertexParam(DIMENSION_NAME) int dimension,
                              @LoadVertexParam(INDEX_NAME) int index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    protected GenericTensor<T> op(GenericTensor<T> input) {
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

package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class IntegerSliceVertex extends IntegerUnaryOpVertex {

    private final static String DIMENSION_NAME = "dimension";
    private final static String INDEX_NAME = "index";

    private final int dimension;
    private final int index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public IntegerSliceVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                              @LoadVertexParam(DIMENSION_NAME) int dimension,
                              @LoadVertexParam(INDEX_NAME) int index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return value.slice(dimension, index);
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

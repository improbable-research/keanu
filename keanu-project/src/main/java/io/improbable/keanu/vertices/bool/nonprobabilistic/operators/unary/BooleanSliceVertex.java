package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;

import static io.improbable.keanu.tensor.TensorShape.removeDimension;

public class BooleanSliceVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    private final static String DIMENSION_NAME = "dimension";
    private final static String INDEX_NAME = "index";

    private final int dimension;
    private final long index;

    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    @ExportVertexToPythonBindings
    public BooleanSliceVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor> inputVertex,
                              @LoadVertexParam(DIMENSION_NAME) int dimension,
                              @LoadVertexParam(INDEX_NAME) long index) {
        super(removeDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return value.slice(dimension, index);
    }

    @SaveVertexParam(DIMENSION_NAME)
    public int getDimension() {
        return dimension;
    }

    @SaveVertexParam(INDEX_NAME)
    public long getIndex() {
        return index;
    }
}

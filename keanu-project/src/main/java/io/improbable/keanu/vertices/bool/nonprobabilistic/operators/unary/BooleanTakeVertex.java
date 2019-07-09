package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;

public class BooleanTakeVertex extends BooleanUnaryOpVertex<BooleanTensor> {

    private static final String INDEX_NAME = "index";
    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index       the index to extract at
     */
    @ExportVertexToPythonBindings
    public BooleanTakeVertex(@LoadVertexParam(INPUT_NAME) IVertex<BooleanTensor> inputVertex,
                             @LoadVertexParam(INDEX_NAME) long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }

    @Override
    protected BooleanTensor op(BooleanTensor value) {
        return BooleanTensor.scalar(value.getValue(index));
    }

    @SaveVertexParam(INDEX_NAME)
    public long[] getIndex() {
        return index;
    }
}

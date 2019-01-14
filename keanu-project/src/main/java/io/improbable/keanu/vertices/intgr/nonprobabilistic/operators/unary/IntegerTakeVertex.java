package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerTakeVertex extends IntegerUnaryOpVertex {

    private static final String INDEX_NAME = "index";
    private final long[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    @ExportVertexToPythonBindings
    public IntegerTakeVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                             @LoadVertexParam(INDEX_NAME) long... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        this.index = index;
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }

    @Override
    protected IntegerTensor op(IntegerTensor value) {
        return IntegerTensor.scalar(value.getValue(index));
    }

    @SaveVertexParam(INDEX_NAME)
    public long[] getIndex() {
        return index;
    }
}

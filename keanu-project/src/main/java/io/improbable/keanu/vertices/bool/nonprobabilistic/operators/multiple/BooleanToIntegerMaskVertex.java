package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class BooleanToIntegerMaskVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, NonProbabilistic<IntegerTensor> {
    private static final String INPUT_NAME = "inputName";
    private final BooleanVertex inputVertex;

    @ExportVertexToPythonBindings
    public BooleanToIntegerMaskVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex) {
        this.inputVertex = inputVertex;
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue().toIntegerMask();
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return this.inputVertex;
    }
}

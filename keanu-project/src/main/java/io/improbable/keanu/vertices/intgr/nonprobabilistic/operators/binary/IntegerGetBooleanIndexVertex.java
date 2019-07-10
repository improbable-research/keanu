package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;


public class IntegerGetBooleanIndexVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, NonProbabilistic<IntegerTensor>, VertexBinaryOp<IntegerVertex, BooleanVertex> {

    private static final String INDICES = "indices";
    private static final String INPUT_NAME = "inputVertex";

    private final IntegerVertex inputVertex;
    private final BooleanVertex indices;

    @ExportVertexToPythonBindings
    public IntegerGetBooleanIndexVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex,
                                        @LoadVertexParam(INDICES) BooleanVertex indices) {
        super(indices.getShape());
        this.inputVertex = inputVertex;
        this.indices = indices;
        setParents(indices, inputVertex);
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue().get(indices.getValue());
    }

    @Override
    @SaveVertexParam(INPUT_NAME)
    public IntegerVertex getLeft() {
        return inputVertex;
    }

    @Override
    @SaveVertexParam(INDICES)
    public BooleanVertex getRight() {
        return indices;
    }
}

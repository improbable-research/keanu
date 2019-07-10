package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexUnaryOp;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public class IntegerUnaryOpLambda extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, NonProbabilistic<IntegerTensor>, NonSaveableVertex, VertexUnaryOp<IntegerVertex> {

    protected static final String INPUT_NAME = "inputVertex";

    protected final IntegerVertex inputVertex;
    protected final Function<IntegerTensor, IntegerTensor> op;

    public IntegerUnaryOpLambda(long[] shape, IntegerVertex inputVertex, Function<IntegerTensor, IntegerTensor> op) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
    }

    public IntegerUnaryOpLambda(IntegerVertex inputVertex, Function<IntegerTensor, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(inputVertex.getValue());
    }

    @SaveVertexParam(INPUT_NAME)
    public IntegerVertex getInputVertex() {
        return inputVertex;
    }
}

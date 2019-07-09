package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public class IntegerUnaryOpLambda<IN> extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    protected final IVertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(long[] shape, IVertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
    }

    public IntegerUnaryOpLambda(IVertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(inputVertex.getValue());
    }
}

package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public class IntegerApplyVertex extends VertexImpl<IntegerTensor> implements IntegerVertex, NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    protected final IntegerVertex inputVertex;
    protected final Function<Integer, Integer> op;

    public IntegerApplyVertex(long[] shape, IntegerVertex inputVertex, Function<Integer, Integer> op) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
    }

    public IntegerApplyVertex(IntegerVertex inputVertex, Function<Integer, Integer> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue().apply(op);
    }
}

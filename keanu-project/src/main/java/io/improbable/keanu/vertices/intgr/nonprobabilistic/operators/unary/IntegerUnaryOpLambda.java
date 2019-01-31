package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public class IntegerUnaryOpLambda<IN> extends IntegerVertex implements NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(long[] shape, Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
    }

    public IntegerUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(inputVertex.getValue());
    }
}

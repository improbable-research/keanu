package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

import java.util.function.Function;

public class IntegerUnaryOpLambda<IN> extends NonProbabilisticInteger {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, Integer> op;

    public IntegerUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, Integer> op) {
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
    }

    @Override
    public Integer sample() {
        return op.apply(inputVertex.sample());
    }

    @Override
    public Integer getDerivedValue() {
        return op.apply(inputVertex.getValue());
    }
}

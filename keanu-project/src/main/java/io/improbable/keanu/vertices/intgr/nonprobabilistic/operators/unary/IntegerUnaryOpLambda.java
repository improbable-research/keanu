package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public class IntegerUnaryOpLambda<IN> extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(long[] shape, Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public IntegerUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(inputVertex.getValue());
    }
}

package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import java.util.function.Function;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerUnaryOpLambda<IN> extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
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
    public void calculate() {
        setValue(op.apply(inputVertex.getValue()));
    }
}

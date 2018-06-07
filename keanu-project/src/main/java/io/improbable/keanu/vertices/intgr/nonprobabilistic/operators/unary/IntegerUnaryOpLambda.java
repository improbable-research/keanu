package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

import java.util.function.Function;

public class IntegerUnaryOpLambda<IN> extends NonProbabilisticInteger {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public IntegerTensor getDerivedValue() {
        return op.apply(inputVertex.getValue());
    }
}

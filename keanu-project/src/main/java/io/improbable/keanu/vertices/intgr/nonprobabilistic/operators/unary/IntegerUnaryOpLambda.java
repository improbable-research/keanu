package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import java.util.function.Function;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class IntegerUnaryOpLambda<IN> extends IntegerVertex {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, IntegerTensor> op;

    public IntegerUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, IntegerTensor> op) {
        super(new NonProbabilisticValueUpdater<>(v -> ((IntegerUnaryOpLambda<IN>) v).op.apply(inputVertex.getValue())));
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

}

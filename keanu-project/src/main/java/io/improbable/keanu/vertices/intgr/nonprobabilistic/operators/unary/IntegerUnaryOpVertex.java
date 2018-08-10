package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import java.util.function.Function;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class IntegerUnaryOpVertex extends IntegerVertex {

    protected final IntegerVertex inputVertex;
    private final Function<IntegerTensor, IntegerTensor> op;

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param inputVertex the input vertex
     * @param op          operation used to sample
     */
    public IntegerUnaryOpVertex(IntegerVertex inputVertex, Function<IntegerTensor, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);

    }

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param shape       the shape of the tensor
     * @param inputVertex the input vertex
     * @param op          operation used to sample
     */
    public IntegerUnaryOpVertex(int[] shape, IntegerVertex inputVertex, Function<IntegerTensor, IntegerTensor> op) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(inputVertex.getValue())));
        this.inputVertex = inputVertex;
        this.op = op;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }
}

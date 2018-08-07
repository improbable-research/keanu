package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class IntegerUnaryOpVertex extends IntegerVertex {

    protected final IntegerVertex inputVertex;

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param shape the shape of the resulting vertex
     * @param inputVertex a vertex
     */
    public IntegerUnaryOpVertex(int[] shape, IntegerVertex inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((IntegerUnaryOpVertex) v).op(inputVertex.getValue())));
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected abstract IntegerTensor op(IntegerTensor a);

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class DoubleUnaryOpVertex extends DoubleVertex {

    protected final DoubleVertex inputVertex;

    /**
     * A vertex that performs a user defined operation on a singe input vertex
     *
     * @param shape the shape of the resulting vertex
     * @param inputVertex a vertex
     */
    public DoubleUnaryOpVertex(int[] shape, DoubleVertex inputVertex) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((DoubleUnaryOpVertex) v).op(inputVertex.getValue())),
            Observable.observableTypeFor(DoubleUnaryOpVertex.class)
        );
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(inputVertex.sample(random));
    }

    protected abstract DoubleTensor op(DoubleTensor a);

}

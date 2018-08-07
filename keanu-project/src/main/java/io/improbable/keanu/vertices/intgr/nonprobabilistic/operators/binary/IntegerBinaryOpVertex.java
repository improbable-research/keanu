package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class IntegerBinaryOpVertex extends IntegerVertex {

    protected final IntegerVertex a;
    protected final IntegerVertex b;

    public IntegerBinaryOpVertex(int[] shape, IntegerVertex a, IntegerVertex b) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((IntegerBinaryOpVertex) v).op(a.getValue(), b.getValue())),
            Observable.observableTypeFor(IntegerBinaryOpVertex.class)
        );
        this.a = a;
        this.b = b;
        setParents(a, b);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    protected abstract IntegerTensor op(IntegerTensor a, IntegerTensor b);
}

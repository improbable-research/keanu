package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;


import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public abstract class DoubleBinaryOpVertex extends NonProbabilisticDouble {

    protected final DoubleVertex a;
    protected final DoubleVertex b;

    public DoubleBinaryOpVertex(DoubleVertex a, DoubleVertex b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public Double sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public Double getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract Double op(Double a, Double b);
}

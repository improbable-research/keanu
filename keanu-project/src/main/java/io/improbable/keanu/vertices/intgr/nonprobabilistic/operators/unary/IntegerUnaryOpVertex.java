package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.NonProbabilisticInteger;

import java.util.Random;

public abstract class IntegerUnaryOpVertex extends NonProbabilisticInteger {

    protected final IntegerVertex inputVertex;

    public IntegerUnaryOpVertex(IntegerVertex inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Integer sample(Random random) {
        return op(inputVertex.sample(random));
    }

    public Integer getDerivedValue() {
        return op(inputVertex.getValue());
    }

    protected abstract Integer op(Integer a);

}

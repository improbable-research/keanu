package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.Function;

public abstract class BoolUnaryOpLambda<A> extends BoolUnaryOpVertex<A> {

    private final Function<A, BooleanTensor> boolOp;

    public BoolUnaryOpLambda(Vertex<A> a, Function<A, BooleanTensor> boolOp) {
        super(a);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(A a) {
        return boolOp.apply(a);
    }
}

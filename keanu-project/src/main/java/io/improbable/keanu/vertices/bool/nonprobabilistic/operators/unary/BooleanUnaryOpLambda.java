package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.IVertex;

import java.util.function.Function;

public abstract class BooleanUnaryOpLambda<A extends Tensor> extends BooleanUnaryOpVertex<A> {

    private final Function<A, BooleanTensor> boolOp;

    public BooleanUnaryOpLambda(long[] shape, IVertex<A> a, Function<A, BooleanTensor> boolOp) {
        super(shape, a);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(A a) {
        return boolOp.apply(a);
    }
}

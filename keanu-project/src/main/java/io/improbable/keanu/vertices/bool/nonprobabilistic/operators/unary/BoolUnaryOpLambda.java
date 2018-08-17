package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.Function;

public abstract class BoolUnaryOpLambda<A extends Tensor> extends BoolUnaryOpVertex<A> {

    private final Function<A, BooleanTensor> boolOp;

    public BoolUnaryOpLambda(int[] shape, Vertex<A> a, Function<A, BooleanTensor> boolOp) {
        super(shape, a);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(A a) {
        return boolOp.apply(a);
    }
}

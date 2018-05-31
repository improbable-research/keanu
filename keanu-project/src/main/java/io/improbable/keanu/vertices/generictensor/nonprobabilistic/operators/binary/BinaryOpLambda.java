package io.improbable.keanu.vertices.generictensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public class BinaryOpLambda<A, B, C, TA extends Tensor<A>, TB extends Tensor<B>, TC extends Tensor<C>> extends BinaryOpVertex<A, B, C, TA, TB, TC> {

    private BiFunction<TA, TB, TC> op;

    public BinaryOpLambda(Vertex<TA> a,
                          Vertex<TB> b,
                          BiFunction<TA, TB, TC> op) {
        super(a, b);
        this.op = op;
    }

    @Override
    protected TC op(TA a, TB b) {
        return op.apply(a, b);
    }

}

package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.Function;

public class UnaryOpLambda<IN_TENSOR extends Tensor, OUT_TENSOR extends Tensor> extends UnaryOpVertex<IN_TENSOR, OUT_TENSOR> {

    private Function<IN_TENSOR, OUT_TENSOR> op;

    public UnaryOpLambda(Vertex<IN_TENSOR> inputVertex, Function<IN_TENSOR, OUT_TENSOR> op) {
        super(inputVertex);
        this.op = op;
    }

    @Override
    protected OUT_TENSOR op(IN_TENSOR input) {
        return op.apply(input);
    }

}
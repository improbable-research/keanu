package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;

import java.util.function.Function;

public class UnaryOpLambda<IN, OUT> extends UnaryOpVertex<IN, OUT> implements NonSaveableVertex {

    private Function<IN, OUT> op;

    public UnaryOpLambda(long[] shape, IVertex<IN> inputVertex, Function<IN, OUT> op) {
        super(shape, inputVertex);
        this.op = op;
    }

    @Override
    protected OUT op(IN input) {
        return op.apply(input);
    }

}
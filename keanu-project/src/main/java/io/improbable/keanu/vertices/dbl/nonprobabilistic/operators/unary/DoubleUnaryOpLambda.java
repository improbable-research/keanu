package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.function.Function;
import java.util.function.Supplier;

public class DoubleUnaryOpLambda<IN> extends NonProbabilisticDouble {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, Double> op;
    protected final Supplier<DualNumber> dualNumberSupplier;

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, Double> op, Supplier<DualNumber> dualNumberSupplier) {
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualNumberSupplier = dualNumberSupplier;
        setParents(inputVertex);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, Double> op) {
        this(inputVertex, op, null);
    }

    @Override
    public Double sample() {
        return op.apply(inputVertex.sample());
    }

    @Override
    public Double getDerivedValue() {
        return op.apply(inputVertex.getValue());
    }

    @Override
    public DualNumber getDualNumber() {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.get();
        }

        throw new UnsupportedOperationException();
    }
}



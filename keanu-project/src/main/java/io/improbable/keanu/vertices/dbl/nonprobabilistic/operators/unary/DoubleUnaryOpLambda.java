package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;
import java.util.Random;
import java.util.function.Function;

public class DoubleUnaryOpLambda<IN> extends NonProbabilisticDouble {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, Double> op;
    protected final Function<Map<Vertex, DualNumber>, DualNumber> dualNumberSupplier;

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, Double> op, Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualNumberSupplier = dualNumberCalculation;
        setParents(inputVertex);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, Double> op) {
        this(inputVertex, op, null);
    }

    @Override
    public Double sample(Random random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public Double getDerivedValue() {
        return op.apply(inputVertex.getValue());
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



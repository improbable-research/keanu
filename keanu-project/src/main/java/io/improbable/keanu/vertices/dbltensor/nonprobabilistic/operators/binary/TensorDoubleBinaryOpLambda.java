package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class TensorDoubleBinaryOpLambda<A, B> extends NonProbabilisticDoubleTensor {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation;

    public TensorDoubleBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op, Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation) {
        this.a = a;
        this.b = b;
        this.op = op;
        this.dualNumberCalculation = dualNumberCalculation;
        setParents(a, b);
    }

    public TensorDoubleBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(a, b, op, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op.apply(a.getValue(), b.getValue());
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        if (dualNumberCalculation != null) {
            return dualNumberCalculation.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



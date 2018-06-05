package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class TensorDoubleBinaryOpLambda<A, B> extends NonProbabilisticDoubleTensor {

    protected final Vertex<A> a;
    protected final Vertex<B> b;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation;

    public TensorDoubleBinaryOpLambda(int[] shape,
                                      Vertex<A> a,
                                      Vertex<B> b,
                                      BiFunction<A, B, DoubleTensor> op,
                                      Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation) {
        this.a = a;
        this.b = b;
        this.op = op;
        this.dualNumberCalculation = dualNumberCalculation;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorDoubleBinaryOpLambda(int[] shape, Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(shape, a, b, op, null);
    }

    public TensorDoubleBinaryOpLambda(Vertex<A> a,
                                      Vertex<B> b,
                                      BiFunction<A, B, DoubleTensor> op,
                                      Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op, dualNumberCalculation);
    }

    public TensorDoubleBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op, null);
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



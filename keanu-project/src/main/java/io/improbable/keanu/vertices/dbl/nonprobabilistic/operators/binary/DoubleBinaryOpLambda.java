package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class DoubleBinaryOpLambda<A, B> extends NonProbabilisticDouble {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation;

    public DoubleBinaryOpLambda(int[] shape,
                                Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this.left = left;
        this.right = right;
        this.op = op;
        this.dualNumberCalculation = dualNumberCalculation;
        setParents(left, right);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleBinaryOpLambda(int[] shape, Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(shape, left, right, op, null);
    }

    public DoubleBinaryOpLambda(Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right, op, dualNumberCalculation);
    }

    public DoubleBinaryOpLambda(Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right, op, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(left.sample(random), right.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op.apply(left.getValue(), right.getValue());
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (dualNumberCalculation != null) {
            return dualNumberCalculation.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



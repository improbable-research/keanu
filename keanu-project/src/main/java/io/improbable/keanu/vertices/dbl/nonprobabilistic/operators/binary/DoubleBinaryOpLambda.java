package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class DoubleBinaryOpLambda<A, B> extends DoubleVertex implements Differentiable {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation;

    public DoubleBinaryOpLambda(int[] shape,
                                Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((DoubleBinaryOpLambda<A, B>) v).op.apply(left.getValue(), right.getValue())),
            Observable.observableTypeFor(DoubleBinaryOpLambda.class)
        );
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
                                Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
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
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (dualNumberCalculation != null) {
            return dualNumberCalculation.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class DoubleBinaryOpLambda<A, B> extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda;
    protected final Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda;

    public DoubleBinaryOpLambda(long[] shape,
                                Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                                Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        this.left = left;
        this.right = right;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(left, right);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleBinaryOpLambda(long[] shape, Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(shape, left, right, op, null, null);
    }

    public DoubleBinaryOpLambda(Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                                Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleBinaryOpLambda(Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right, op, null, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(left.sample(random), right.sample(random));
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.apply(derivativeOfParentsWithRespectToInputs);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        return reverseModeAutoDiffLambda.apply(derivativeOfOutputsWithRespectToSelf);
    }
}



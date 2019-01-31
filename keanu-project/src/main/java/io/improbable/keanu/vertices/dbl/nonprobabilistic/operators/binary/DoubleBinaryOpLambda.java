package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleBinaryOpLambda<A, B> extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda;
    protected final Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda;

    public DoubleBinaryOpLambda(long[] shape,
                                Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        this.left = left;
        this.right = right;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(left, right);
    }

    public DoubleBinaryOpLambda(long[] shape, Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(shape, left, right, op, null, null);
    }

    public DoubleBinaryOpLambda(Vertex<A> left,
                                Vertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleBinaryOpLambda(Vertex<A> left, Vertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, null, null);
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.apply(derivativeOfParentsWithRespectToInput);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleBinaryOpLambda<A, B> extends VertexImpl<DoubleTensor> implements DoubleVertex,  Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    protected final IVertex<A> left;
    protected final IVertex<B> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda;
    protected final Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda;

    public DoubleBinaryOpLambda(long[] shape,
                                IVertex<A> left,
                                IVertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        this.left = left;
        this.right = right;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(left, right);
    }

    public DoubleBinaryOpLambda(long[] shape, IVertex<A> left, IVertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(shape, left, right, op, null, null);
    }

    public DoubleBinaryOpLambda(IVertex<A> left,
                                IVertex<B> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleBinaryOpLambda(IVertex<A> left, IVertex<B> right, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, null, null);
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<IVertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.apply(derivativeOfParentsWithRespectToInput);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<IVertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
    }
}

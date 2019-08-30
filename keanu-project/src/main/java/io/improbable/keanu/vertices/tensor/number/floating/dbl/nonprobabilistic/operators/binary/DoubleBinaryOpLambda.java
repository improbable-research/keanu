package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;

import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleBinaryOpLambda<A, B>
    extends VertexImpl<DoubleTensor, DoubleVertex>
    implements DoubleVertex, Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    protected final Vertex<A, ?> left;
    protected final Vertex<B, ?> right;
    protected final BiFunction<A, B, DoubleTensor> op;
    protected final Function<Map<Vertex, ForwardModePartialDerivative>, ForwardModePartialDerivative> forwardModeAutoDiffLambda;
    protected final Function<ReverseModePartialDerivative, Map<Vertex, ReverseModePartialDerivative>> reverseModeAutoDiffLambda;

    public DoubleBinaryOpLambda(long[] shape,
                                Vertex<A, ?> left,
                                Vertex<B, ?> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, ForwardModePartialDerivative>, ForwardModePartialDerivative> forwardModeAutoDiffLambda,
                                Function<ReverseModePartialDerivative, Map<Vertex, ReverseModePartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        this.left = left;
        this.right = right;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(left, right);
    }

    public DoubleBinaryOpLambda(long[] shape, Vertex<A, ?> left, Vertex<B, ?> right, BiFunction<A, B, DoubleTensor> op) {
        this(shape, left, right, op, null, null);
    }

    public DoubleBinaryOpLambda(Vertex<A, ?> left,
                                Vertex<B, ?> right,
                                BiFunction<A, B, DoubleTensor> op,
                                Function<Map<Vertex, ForwardModePartialDerivative>, ForwardModePartialDerivative> forwardModeAutoDiffLambda,
                                Function<ReverseModePartialDerivative, Map<Vertex, ReverseModePartialDerivative>> reverseModeAutoDiffLambda) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleBinaryOpLambda(Vertex<A, ?> left, Vertex<B, ?> right, BiFunction<A, B, DoubleTensor> op) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op, null, null);
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }

    @Override
    public ForwardModePartialDerivative forwardModeAutoDifferentiation(Map<Vertex, ForwardModePartialDerivative> derivativeOfParentsWithRespectToInput) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.apply(derivativeOfParentsWithRespectToInput);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, ReverseModePartialDerivative> reverseModeAutoDifferentiation(ReverseModePartialDerivative derivativeOfOutputWithRespectToSelf) {
        return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
    }
}

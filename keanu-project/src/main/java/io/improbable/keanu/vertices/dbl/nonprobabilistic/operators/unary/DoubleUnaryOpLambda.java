package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;
import java.util.function.Function;

public class DoubleUnaryOpLambda<IN> extends VertexImpl<DoubleTensor> implements DoubleVertex, Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final IVertex<IN> inputVertex;
    private final Function<IN, DoubleTensor> op;
    private final Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda;
    private final Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda;

    public DoubleUnaryOpLambda(long[] shape, IVertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                               Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(inputVertex);
    }

    public DoubleUnaryOpLambda(long[] shape, IVertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null, null);
    }

    public DoubleUnaryOpLambda(IVertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                               Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        this(inputVertex.getShape(), inputVertex, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleUnaryOpLambda(IVertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(inputVertex.getShape(), inputVertex, op, null, null);
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(inputVertex.getValue());
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
        if (reverseModeAutoDiffLambda != null) {
            return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
        }

        throw new UnsupportedOperationException();
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Map;
import java.util.function.Function;

public class DoubleUnaryOpLambda<IN> extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, NonSaveableVertex {

    private final Vertex<IN> inputVertex;
    private final Function<IN, DoubleTensor> op;
    private final Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda;
    private final Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda;

    public DoubleUnaryOpLambda(long[] shape, Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                               Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        super(shape);
        this.inputVertex = inputVertex;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(inputVertex);
    }

    public DoubleUnaryOpLambda(long[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null, null);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                               Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        this(inputVertex.getShape(), inputVertex, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(inputVertex.getShape(), inputVertex, op, null, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public DoubleTensor calculate() {
        return op.apply(inputVertex.getValue());
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
        if (reverseModeAutoDiffLambda != null) {
            return reverseModeAutoDiffLambda.apply(derivativeOfOutputWithRespectToSelf);
        }

        throw new UnsupportedOperationException();
    }
}

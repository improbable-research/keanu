package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class DoubleUnaryOpLambda<IN> extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

    private final Vertex<IN> inputVertex;
    private final Function<IN, DoubleTensor> op;
    private final Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda;
    private final Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda;

    public DoubleUnaryOpLambda(long[] shape, Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        this.inputVertex = inputVertex;
        this.op = op;
        this.forwardModeAutoDiffLambda = forwardModeAutoDiffLambda;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleUnaryOpLambda(long[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null, null);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, PartialDerivatives>, PartialDerivatives> forwardModeAutoDiffLambda,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
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
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        if (forwardModeAutoDiffLambda != null) {
            return forwardModeAutoDiffLambda.apply(derivativeOfParentsWithRespectToInputs);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        if (reverseModeAutoDiffLambda != null) {
            return reverseModeAutoDiffLambda.apply(derivativeOfOutputsWithRespectToSelf);
        }

        throw new UnsupportedOperationException();
    }
}

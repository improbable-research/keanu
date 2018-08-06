package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.NonProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;
import java.util.function.Function;

public class DoubleUnaryOpLambda<IN> extends NonProbabilisticDouble {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, DoubleTensor> op;
    protected final Function<Map<Vertex, DualNumber>, DualNumber> dualNumberSupplier;
    protected final Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda;

    public DoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualNumberSupplier = dualNumberCalculation;
        this.reverseModeAutoDiffLambda = reverseModeAutoDiffLambda;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null, null);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex,
                               Function<IN, DoubleTensor> op,
                               Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation,
                               Function<PartialDerivatives, Map<Vertex, PartialDerivatives>> reverseModeAutoDiffLambda) {
        this(inputVertex.getShape(), inputVertex, op, dualNumberCalculation, reverseModeAutoDiffLambda);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(inputVertex.getShape(), inputVertex, op, null, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return op.apply(inputVertex.getValue());
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }

    @Override
    protected Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        if (reverseModeAutoDiffLambda != null) {
            return reverseModeAutoDiffLambda.apply(derivativeOfOutputsWithRespectToSelf);
        }

        throw new UnsupportedOperationException();
    }
}



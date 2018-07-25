package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;
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

public class DoubleUnaryOpLambda<IN> extends DoubleVertex implements Differentiable {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, DoubleTensor> op;
    protected final Function<Map<IVertex, DualNumber>, DualNumber> dualNumberSupplier;

    public DoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op, Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((DoubleUnaryOpLambda<IN>) v).op.apply(inputVertex.getValue())),
            Observable.observableTypeFor(DoubleUnaryOpLambda.class)
        );
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualNumberSupplier = dualNumberCalculation;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public DoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op, Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        this(inputVertex.getShape(), inputVertex, op, dualNumberCalculation);
    }

    public DoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(inputVertex.getShape(), inputVertex, op, null);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



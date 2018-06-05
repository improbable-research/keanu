package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.NonProbabilisticDoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Map;
import java.util.function.Function;

public class TensorDoubleUnaryOpLambda<IN> extends NonProbabilisticDoubleTensor {

    protected final Vertex<IN> inputVertex;
    protected final Function<IN, DoubleTensor> op;
    protected final Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberSupplier;

    public TensorDoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op, Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation) {
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualNumberSupplier = dualNumberCalculation;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    public TensorDoubleUnaryOpLambda(int[] shape, Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(shape, inputVertex, op, null);
    }

    public TensorDoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op, Function<Map<Vertex, TensorDualNumber>, TensorDualNumber> dualNumberCalculation) {
        this(inputVertex.getShape(), inputVertex, op, dualNumberCalculation);
    }

    public TensorDoubleUnaryOpLambda(Vertex<IN> inputVertex, Function<IN, DoubleTensor> op) {
        this(inputVertex.getShape(), inputVertex, op, null);
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
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        if (dualNumberSupplier != null) {
            return dualNumberSupplier.apply(dualNumbers);
        }

        throw new UnsupportedOperationException();
    }
}



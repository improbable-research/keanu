package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class DoubleUnaryOpVertex extends DoubleVertex {

    protected final DoubleVertex inputVertex;
    private final Function<DoubleTensor,DoubleTensor> op;
    private final Function<DualNumber,DualNumber> dualOp;

    /**
     * A vertex that performs a user defined operation on a single input vertex
     * @param inputVertex the input vertex
     * @param op operation used to sample
     */
    public DoubleUnaryOpVertex(DoubleVertex inputVertex, Function<DoubleTensor,DoubleTensor> op) {
        this(inputVertex, op, null);
    }

    /**
     * A vertex that performs a user defined operation on a single input vertex
     * @param inputVertex the input vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleUnaryOpVertex(
        DoubleVertex inputVertex,
        Function<DoubleTensor,DoubleTensor> op,
        Function<DualNumber,DualNumber> dualOp) {
        this(inputVertex.getShape(), inputVertex, op, dualOp);
    }

    /**
     * A vertex that performs a user defined operation on a single input vertex
     * @param shape the shape of the tensor
     * @param inputVertex the input vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleUnaryOpVertex(
        int[] shape,
        DoubleVertex inputVertex,
        Function<DoubleTensor,DoubleTensor> op,
        Function<DualNumber,DualNumber> dualOp) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(inputVertex.getValue())),
            Observable.observableTypeFor(DoubleUnaryOpVertex.class)
        );
        this.inputVertex = inputVertex;
        this.op = op;
        this.dualOp = dualOp;
        setParents(inputVertex);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(inputVertex.sample(random));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (dualOp == null) {
            return super.calculateDualNumber(dualNumbers);
        } else {
            return dualOp.apply(dualNumbers.get(inputVertex));
        }
    }
}

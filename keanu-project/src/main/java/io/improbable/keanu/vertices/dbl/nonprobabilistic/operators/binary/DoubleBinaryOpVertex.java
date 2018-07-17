package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;
import java.util.function.BinaryOperator;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class DoubleBinaryOpVertex extends DoubleVertex {

    private final DoubleVertex a;
    private final DoubleVertex b;
    private final BinaryOperator<DoubleTensor> op;
    private final BinaryOperator<DualNumber> dualOp;

    /**
     * A vertex that performs a user defined operation on two input vertices
     * @param a first input vertex
     * @param b second input vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleBinaryOpVertex(
        DoubleVertex a, DoubleVertex b,
        BinaryOperator<DoubleTensor> op, BinaryOperator<DualNumber> dualOp) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()),
            a, b, op, dualOp);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     * @param shape the shape of the tensor
     * @param a first input vertex
     * @param b second input vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleBinaryOpVertex(
        int[] shape,
        DoubleVertex a, DoubleVertex b,
        BinaryOperator<DoubleTensor> op, BinaryOperator<DualNumber> dualOp) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(a.getValue(), b.getValue())),
            Observable.observableTypeFor(DoubleBinaryOpVertex.class)
        );
        this.a = a;
        this.b = b;
        this.op = op;
        this.dualOp = dualOp;
        setParents(a, b);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        return dualOp.apply(dualNumbers.get(a), dualNumbers.get(b));
    }
}

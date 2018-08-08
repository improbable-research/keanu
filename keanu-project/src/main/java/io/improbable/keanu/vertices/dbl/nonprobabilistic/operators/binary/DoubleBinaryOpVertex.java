package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;
import java.util.function.BinaryOperator;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class DoubleBinaryOpVertex extends DoubleVertex {

    private final DoubleVertex left;
    private final DoubleVertex right;
    private final BinaryOperator<DoubleTensor> op;
    private final BinaryOperator<DualNumber> dualOp;

    /**
     * A vertex that performs a user defined operation on two vertices
     * @param left  a vertex
     * @param right a vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleBinaryOpVertex(
        DoubleVertex left, DoubleVertex right,
        BinaryOperator<DoubleTensor> op, BinaryOperator<DualNumber> dualOp) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()),
            left, right, op, dualOp);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     * @param shape the shape of the tensor
     * @param left first input vertex
     * @param right second input vertex
     * @param op operation used to sample
     * @param dualOp operation used to calculate Dual
     */
    public DoubleBinaryOpVertex(
        int[] shape,
        DoubleVertex left, DoubleVertex right,
        BinaryOperator<DoubleTensor> op, BinaryOperator<DualNumber> dualOp) {
        super(new NonProbabilisticValueUpdater<>(v -> op.apply(left.getValue(), right.getValue())));
        this.left = left;
        this.right = right;
        this.op = op;
        this.dualOp = dualOp;
        setParents(left, right);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op.apply(left.sample(random), right.sample(random));
    }

    public DoubleVertex getLeft() {
        return left;
    }

    public DoubleVertex getRight() {
        return right;
    }
    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualOp.apply(dualNumbers.get(left), dualNumbers.get(right));
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class DoubleBinaryOpVertex extends DoubleVertex {

    private final DoubleVertex left;
    private final DoubleVertex right;

    /**
     * A vertex that performs a user defined operation on two vertices
     *
     * @param left  a vertex
     * @param right a vertex
     */
    public DoubleBinaryOpVertex(
        DoubleVertex left, DoubleVertex right) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()),
            left, right);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param shape the shape of the tensor
     * @param left  first input vertex
     * @param right second input vertex
     */
    public DoubleBinaryOpVertex(
        int[] shape,
        DoubleVertex left, DoubleVertex right) {
        super(new NonProbabilisticValueUpdater<>(v -> ((DoubleBinaryOpVertex) v).op(left.getValue(), right.getValue())));
        this.left = left;
        this.right = right;
        setParents(left, right);
        setValue(DoubleTensor.placeHolder(shape));
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return op(left.sample(random), right.sample(random));
    }

    public DoubleVertex getLeft() {
        return left;
    }

    public DoubleVertex getRight() {
        return right;
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualOp(dualNumbers.get(left), dualNumbers.get(right));
    }

    protected abstract DoubleTensor op(DoubleTensor l, DoubleTensor r);

    protected abstract DualNumber dualOp(DualNumber l, DualNumber r);
}

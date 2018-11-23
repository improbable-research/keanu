package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public abstract class IntegerBinaryOpVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    protected final IntegerVertex left;
    protected final IntegerVertex right;
    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param left first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(IntegerVertex left, IntegerVertex right) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param shape the shape of the tensor
     * @param left  first input vertex
     * @param right second input vertex
     */
    public IntegerBinaryOpVertex(long[] shape, IntegerVertex left, IntegerVertex right) {
        super(shape);
        this.left = left;
        this.right = right;
        setParents(left, right);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op(left.sample(random), right.sample(random));
    }

    @Override
    public IntegerTensor calculate() {
        return op(left.getValue(), right.getValue());
    }

    protected abstract IntegerTensor op(IntegerTensor l, IntegerTensor r);

    @SaveParentVertex(LEFT_NAME)
    public IntegerVertex getLeft() {
        return left;
    }

    @SaveParentVertex(RIGHT_NAME)
    public IntegerVertex getRight() {
        return right;
    }
}

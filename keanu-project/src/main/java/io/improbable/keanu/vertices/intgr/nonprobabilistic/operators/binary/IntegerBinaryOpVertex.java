package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;


import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public abstract class IntegerBinaryOpVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor>, VertexBinaryOp<IntegerVertex, IntegerVertex> {

    protected final IntegerVertex left;
    protected final IntegerVertex right;

    /**
     * A vertex that performs a user defined operation on two input vertices
     *
     * @param left  first input vertex
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
        this.left = left;
        this.right = right;
        setParents(left, right);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerVertex getLeft() {
        return left;
    }

    @Override
    public IntegerVertex getRight() {
        return right;
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
}

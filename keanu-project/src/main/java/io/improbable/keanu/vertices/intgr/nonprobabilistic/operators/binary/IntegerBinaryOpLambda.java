package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;


public class IntegerBinaryOpLambda extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    protected static final String LEFT_NAME = "left";
    protected static final String RIGHT_NAME = "right";

    protected final IntegerVertex left;
    protected final IntegerVertex right;
    protected final BiFunction<IntegerTensor, IntegerTensor, IntegerTensor> op;

    public IntegerBinaryOpLambda(long[] shape,
                                 IntegerVertex left,
                                 IntegerVertex right,
                                 BiFunction<IntegerTensor, IntegerTensor, IntegerTensor> op) {
        super(shape);
        this.left = left;
        this.right = right;
        this.op = op;
        setParents(left, right);
    }

    public IntegerBinaryOpLambda(IntegerVertex left, IntegerVertex right, BiFunction<IntegerTensor, IntegerTensor, IntegerTensor> op) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right, op);
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }

    @SaveVertexParam(LEFT_NAME)
    public IntegerVertex getLeft() {
        return left;
    }

    @SaveVertexParam(RIGHT_NAME)
    public IntegerVertex getRight() {
        return right;
    }
}

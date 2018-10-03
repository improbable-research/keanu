package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import java.util.function.BiFunction;

public class IntegerBinaryOpLambda<A, B> extends IntegerVertex
        implements NonProbabilistic<IntegerTensor> {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, IntegerTensor> op;

    public IntegerBinaryOpLambda(
            int[] shape, Vertex<A> left, Vertex<B> right, BiFunction<A, B, IntegerTensor> op) {
        this.left = left;
        this.right = right;
        this.op = op;
        setParents(left, right);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public IntegerBinaryOpLambda(
            Vertex<A> left, Vertex<B> right, BiFunction<A, B, IntegerTensor> op) {
        this(
                checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()),
                left,
                right,
                op);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(left.sample(random), right.sample(random));
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }
}

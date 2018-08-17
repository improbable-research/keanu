package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.function.BiFunction;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;


public class IntegerBinaryOpLambda<A, B> extends IntegerVertex {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, IntegerTensor> op;

    public IntegerBinaryOpLambda(int[] shape,
                                 Vertex<A> left,
                                 Vertex<B> right,
                                 BiFunction<A, B, IntegerTensor> op) {
        super(new NonProbabilisticValueUpdater<>(v -> ((IntegerBinaryOpLambda<A, B>) v).op.apply(left.getValue(), right.getValue())));
        this.left = left;
        this.right = right;
        this.op = op;
        setParents(left, right);
        setValue(IntegerTensor.placeHolder(shape));
    }

    public IntegerBinaryOpLambda(Vertex<A> left, Vertex<B> right, BiFunction<A, B, IntegerTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right, op);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(left.sample(random), right.sample(random));
    }
}

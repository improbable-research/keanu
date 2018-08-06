package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;


import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import java.util.function.BinaryOperator;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class IntegerBinaryOpVertex extends IntegerVertex {

    protected final IntegerVertex a;
    protected final IntegerVertex b;
    private final BinaryOperator<IntegerTensor> op;

    /**
     * A vertex that performs a user defined operation on two input vertices
     * @param a first input vertex
     * @param b second input vertex
     * @param op operation used to sample
     */
    public IntegerBinaryOpVertex(IntegerVertex a, IntegerVertex b, BinaryOperator<IntegerTensor> op) {
        this(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, op);
    }

    /**
     * A vertex that performs a user defined operation on two input vertices
     * @param shape the shape of the tensor
     * @param a first input vertex
     * @param b second input vertex
     * @param op operation used to sample
     */
    public IntegerBinaryOpVertex(int[] shape, IntegerVertex a, IntegerVertex b, BinaryOperator<IntegerTensor> op) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(a.getValue(), b.getValue())),
            Observable.observableTypeFor(IntegerBinaryOpVertex.class)
        );
        this.a = a;
        this.b = b;
        this.op = op;
        setParents(a, b);
        setValue(IntegerTensor.placeHolder(shape));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random), b.sample(random));
    }
}

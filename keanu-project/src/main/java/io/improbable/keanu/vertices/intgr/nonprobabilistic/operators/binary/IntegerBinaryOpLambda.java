package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;


public class IntegerBinaryOpLambda<A, B> extends IntegerVertex implements NonProbabilistic<IntegerTensor>, NonSaveableVertex {

    protected final Vertex<A> left;
    protected final Vertex<B> right;
    protected final BiFunction<A, B, IntegerTensor> op;

    public IntegerBinaryOpLambda(long[] shape,
                                 Vertex<A> left,
                                 Vertex<B> right,
                                 BiFunction<A, B, IntegerTensor> op) {
        super(shape);
        this.left = left;
        this.right = right;
        this.op = op;
        setParents(left, right);
    }

    public IntegerBinaryOpLambda(Vertex<A> left, Vertex<B> right, BiFunction<A, B, IntegerTensor> op) {
        this(checkHasOneNonLengthOneShapeOrAllLengthOne(left.getShape(), right.getShape()), left, right, op);
    }

    @Override
    public IntegerTensor calculate() {
        return op.apply(left.getValue(), right.getValue());
    }
}

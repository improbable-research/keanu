package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.bool.BooleanVertex;

import java.util.function.BiFunction;

public class BooleanBinaryOpLambda extends BooleanBinaryOpVertex<BooleanTensor, BooleanTensor>
    implements NonSaveableVertex, VertexBinaryOp<Vertex<BooleanTensor>, Vertex<BooleanTensor>> {

    private final BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> boolOp;

    public BooleanBinaryOpLambda(long[] shape, BooleanVertex a, BooleanVertex b, BiFunction<BooleanTensor, BooleanTensor, BooleanTensor> boolOp) {
        super(shape, a, b);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(BooleanTensor a, BooleanTensor b) {
        return boolOp.apply(a, b);
    }
}

package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexBinaryOp;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;

public class GetBooleanIndexVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, VertexBinaryOp<TensorVertex<T, TENSOR, VERTEX>, BooleanVertex> {

    private final String INPUT_NAME = "inputVertex";
    private final String BOOLEAN_INDEX_NAME = "booleanIndex";
    protected static final String TYPE_NAME = "type";

    private final TensorVertex<T, TENSOR, VERTEX> inputVertex;
    private final BooleanVertex booleanIndex;
    private final Class<?> type;

    public GetBooleanIndexVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                                 @LoadVertexParam(BOOLEAN_INDEX_NAME) BooleanVertex booleanIndex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        this.booleanIndex = booleanIndex;
        this.type = inputVertex.ofType();
        setParents(inputVertex, booleanIndex);
    }

    @Override
    public TENSOR calculate() {
        return inputVertex.getValue().get(booleanIndex.getValue());
    }

    @Override
    public TensorVertex<T, TENSOR, VERTEX> getLeft() {
        return inputVertex;
    }

    @Override
    public BooleanVertex getRight() {
        return booleanIndex;
    }

    @SaveVertexParam(INPUT_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getInputVertex() {
        return this.inputVertex;
    }

    @SaveVertexParam(BOOLEAN_INDEX_NAME)
    public BooleanVertex getBooleanIndex() {
        return this.booleanIndex;
    }

    @Override
    @SaveVertexParam(TYPE_NAME)
    public Class<?> ofType() {
        return type;
    }
}

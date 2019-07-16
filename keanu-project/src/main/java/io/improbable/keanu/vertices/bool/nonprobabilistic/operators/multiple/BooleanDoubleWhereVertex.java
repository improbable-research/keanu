package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class BooleanDoubleWhereVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor> {
    private static final String INPUT_NAME = "inputName";
    private static final String TRUE_VALUE = "trueValue";
    private static final String FALSE_VALUE = "falseValue";

    private final Vertex<BooleanTensor, ?> inputVertex;
    private final Vertex<DoubleTensor, ?> trueValue;
    private final Vertex<DoubleTensor, ?> falseValue;

    @ExportVertexToPythonBindings
    public BooleanDoubleWhereVertex(@LoadVertexParam(INPUT_NAME) Vertex<BooleanTensor, ?> inputVertex,
                                    @LoadVertexParam(TRUE_VALUE) Vertex<DoubleTensor, ?> trueValue,
                                    @LoadVertexParam(FALSE_VALUE) Vertex<DoubleTensor, ?> falseValue) {
        this.inputVertex = inputVertex;
        this.trueValue = trueValue;
        this.falseValue = falseValue;
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().doubleWhere(trueValue.getValue(), falseValue.getValue());
    }

    @SaveVertexParam(INPUT_NAME)
    public Vertex<BooleanTensor, ?> getInputVertex() {
        return this.inputVertex;
    }

    @SaveVertexParam(TRUE_VALUE)
    public Vertex<DoubleTensor, ?> getTrueValue() {
        return this.trueValue;
    }

    @SaveVertexParam(FALSE_VALUE)
    public Vertex<DoubleTensor, ?> getFalseValue() {
        return this.falseValue;
    }
}

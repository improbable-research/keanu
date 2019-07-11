package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class BooleanDoubleWhereVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor> {
    private static final String INPUT_NAME = "inputName";
    private static final String TRUE_VALUE = "trueValue";
    private static final String FALSE_VALUE = "falseValue";

    private final BooleanVertex inputVertex;
    private final DoubleVertex trueValue;
    private final DoubleVertex falseValue;

    @ExportVertexToPythonBindings
    public BooleanDoubleWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                    @LoadVertexParam(TRUE_VALUE) DoubleVertex trueValue,
                                    @LoadVertexParam(FALSE_VALUE) DoubleVertex falseValue) {
        this.inputVertex = inputVertex;
        this.trueValue = trueValue;
        this.falseValue = falseValue;
    }

    @Override
    public DoubleTensor calculate() {
        return inputVertex.getValue().doubleWhere(trueValue.getValue(), falseValue.getValue());
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return this.inputVertex;
    }

    @SaveVertexParam(TRUE_VALUE)
    public DoubleVertex getTrueValue() {
        return this.trueValue;
    }

    @SaveVertexParam(FALSE_VALUE)
    public DoubleVertex getFalseValue() {
        return this.falseValue;
    }
}

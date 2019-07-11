package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class BooleanIntegerWhereVertex extends VertexImpl<IntegerTensor, IntegerVertex> implements IntegerVertex, NonProbabilistic<IntegerTensor> {
    private static final String INPUT_NAME = "inputName";
    private static final String TRUE_VALUE = "trueValue";
    private static final String FALSE_VALUE = "falseValue";

    private final BooleanVertex inputVertex;

    private final IntegerVertex trueValue;

    private final IntegerVertex falseValue;

    @ExportVertexToPythonBindings
    public BooleanIntegerWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                     @LoadVertexParam(TRUE_VALUE) IntegerVertex trueValue,
                                     @LoadVertexParam(FALSE_VALUE) IntegerVertex falseValue) {
        this.inputVertex = inputVertex;
        this.trueValue = trueValue;
        this.falseValue = falseValue;
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue().integerWhere(trueValue.getValue(), falseValue.getValue());
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return this.inputVertex;
    }

    @SaveVertexParam(TRUE_VALUE)
    public IntegerVertex getTrueValue() {
        return this.trueValue;
    }

    @SaveVertexParam(FALSE_VALUE)
    public IntegerVertex getFalseValue() {
        return this.falseValue;
    }
}

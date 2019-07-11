package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class BooleanBooleanWhereVertex extends VertexImpl<BooleanTensor, BooleanVertex> implements BooleanVertex, NonProbabilistic<BooleanTensor> {
    private static final String INPUT_NAME = "inputName";
    private static final String TRUE_VALUE = "trueValue";
    private static final String FALSE_VALUE = "falseValue";

    private final BooleanVertex inputVertex;
    private final BooleanVertex trueValue;
    private final BooleanVertex falseValue;

    @ExportVertexToPythonBindings
    public BooleanBooleanWhereVertex(@LoadVertexParam(INPUT_NAME) BooleanVertex inputVertex,
                                     @LoadVertexParam(TRUE_VALUE) BooleanVertex trueValue,
                                     @LoadVertexParam(FALSE_VALUE) BooleanVertex falseValue) {
        this.inputVertex = inputVertex;
        this.trueValue = trueValue;
        this.falseValue = falseValue;
    }

    @Override
    public BooleanTensor calculate() {
        return inputVertex.getValue().booleanWhere(trueValue.getValue(), falseValue.getValue());
    }

    @SaveVertexParam(INPUT_NAME)
    public BooleanVertex getInputVertex() {
        return this.inputVertex;
    }

    @SaveVertexParam(TRUE_VALUE)
    public BooleanVertex getTrueValue() {
        return this.trueValue;
    }

    @SaveVertexParam(FALSE_VALUE)
    public BooleanVertex getFalseValue() {
        return this.falseValue;
    }
}

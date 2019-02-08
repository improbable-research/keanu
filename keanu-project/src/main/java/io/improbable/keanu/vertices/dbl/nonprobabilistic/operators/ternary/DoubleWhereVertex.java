package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne;

public class DoubleWhereVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final static String CONDITION_NAME = "condition";
    private final static String TRUE_VALUE_NAME = "trueValue";
    private final static String FALSE_VALUE_NAME = "falseValue";

    private final BooleanVertex condition;
    private final DoubleVertex trueValue;
    private final DoubleVertex falseValue;

    @ExportVertexToPythonBindings
    public DoubleWhereVertex(@LoadVertexParam(CONDITION_NAME) BooleanVertex condition,
                             @LoadVertexParam(TRUE_VALUE_NAME) DoubleVertex trueValue,
                             @LoadVertexParam(FALSE_VALUE_NAME) DoubleVertex falseValue) {
        super(condition.getShape());
        checkTensorsMatchNonLengthOneShapeOrAreLengthOne(condition.getShape(), trueValue.getShape(), falseValue.getShape());
        this.condition = condition;
        this.trueValue = trueValue;
        this.falseValue = falseValue;

        setParents(condition, trueValue, falseValue);
    }

    @Override
    public DoubleTensor calculate() {
        return condition.getValue().doubleWhere(trueValue.getValue(), falseValue.getValue());
    }

    @SaveVertexParam(CONDITION_NAME)
    public BooleanVertex getCondition() {
        return condition;
    }

    @SaveVertexParam(TRUE_VALUE_NAME)
    public DoubleVertex getTrueValue() {
        return trueValue;
    }

    @SaveVertexParam(FALSE_VALUE_NAME)
    public DoubleVertex getFalseValue() {
        return falseValue;
    }
}

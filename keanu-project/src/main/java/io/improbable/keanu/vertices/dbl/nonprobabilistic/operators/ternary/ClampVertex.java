package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class ClampVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final static String OPERAND_NAME = "operand";
    private final static String MIN_NAME = "min";
    private final static String MAX_NAME = "max";

    private final DoubleVertex operand;
    private final DoubleVertex min;
    private final DoubleVertex max;

    public ClampVertex(@LoadVertexParam(OPERAND_NAME) DoubleVertex operand,
                       @LoadVertexParam(MIN_NAME) DoubleVertex min,
                       @LoadVertexParam(MAX_NAME) DoubleVertex max) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(operand.getShape(), min.getShape(), max.getShape()));
        this.operand = operand;
        this.min = min;
        this.max = max;
        setParents(operand, min, max);
    }

    @Override
    public DoubleTensor calculate() {
        return operand.getValue().clamp(min.getValue(), max.getValue());
    }

    @SaveVertexParam(OPERAND_NAME)
    public DoubleVertex getOperand() {
        return operand;
    }

    @SaveVertexParam(MIN_NAME)
    public DoubleVertex getMin() {
        return min;
    }

    @SaveVertexParam(MAX_NAME)
    public DoubleVertex getMax() {
        return max;
    }
}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class ClampVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor> {

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

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        PartialDerivative operandPartial = derivativeOfParentsWithRespectToInput.getOrDefault(operand, PartialDerivative.EMPTY);
        PartialDerivative minPartial = derivativeOfParentsWithRespectToInput.getOrDefault(min, PartialDerivative.EMPTY);
        PartialDerivative maxPartial = derivativeOfParentsWithRespectToInput.getOrDefault(max, PartialDerivative.EMPTY);

        DoubleTensor operandValue = operand.getValue();
        DoubleTensor minValue = min.getValue();
        DoubleTensor maxValue = max.getValue();

        DoubleTensor lessThanMinMask = operandValue.getLessThanMask(minValue);
        DoubleTensor greaterThanMaxMask = operandValue.getGreaterThanMask(maxValue);
        DoubleTensor inSupportMask = lessThanMinMask.plus(greaterThanMaxMask).unaryMinusInPlace().plusInPlace(1.);

        return  operandPartial.multiplyAlongOfDimensions(inSupportMask)
            .add(minPartial.multiplyAlongOfDimensions(lessThanMinMask))
            .add(maxPartial.multiplyAlongOfDimensions(greaterThanMaxMask));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();

        DoubleTensor operandValue = operand.getValue();
        DoubleTensor minValue = min.getValue();
        DoubleTensor maxValue = max.getValue();

        DoubleTensor lessThanMinMask = operandValue.getLessThanMask(minValue);
        DoubleTensor greaterThanMaxMask = operandValue.getGreaterThanMask(maxValue);
        DoubleTensor inSupportMask = lessThanMinMask.plus(greaterThanMaxMask).unaryMinusInPlace().plusInPlace(1.);

        partials.put(operand, derivativeOfOutputWithRespectToSelf
            .multiplyAlongOfDimensions(inSupportMask));
        partials.put(min, derivativeOfOutputWithRespectToSelf
            .multiplyAlongOfDimensions(lessThanMinMask));
        partials.put(max, derivativeOfOutputWithRespectToSelf
            .multiplyAlongOfDimensions(greaterThanMaxMask));

        return partials;
    }
}

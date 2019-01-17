package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasOneNonLengthOneShapeOrAllLengthOne;

public class DoubleSetWithMaskVertex extends DoubleVertex implements NonProbabilistic<DoubleTensor> {

    private final static String OPERAND_NAME = "operand";
    private final static String MASK_NAME = "mask";
    private final static String SET_VALUE_NAME = "setValue";

    private final DoubleVertex operand;
    private final DoubleVertex mask;
    private final DoubleVertex setValue;

    @ExportVertexToPythonBindings
    public DoubleSetWithMaskVertex(@LoadVertexParam(OPERAND_NAME) DoubleVertex operand,
                                   @LoadVertexParam(MASK_NAME) DoubleVertex mask,
                                   @LoadVertexParam(SET_VALUE_NAME) DoubleVertex setValue) {
        super(checkHasOneNonLengthOneShapeOrAllLengthOne(operand.getShape(), mask.getShape()));
        this.operand = operand;
        this.mask = mask;
        this.setValue = setValue;
        setParents(operand, mask, setValue);
    }

    @Override
    public DoubleTensor calculate() {
        return operand.getValue().setWithMask(mask.getValue(), setValue.getValue().scalar());
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return this.getValue();
    }

    @SaveVertexParam(OPERAND_NAME)
    public DoubleVertex getOperand() {
        return operand;
    }

    @SaveVertexParam(MASK_NAME)
    public DoubleVertex getMask() {
        return mask;
    }

    @SaveVertexParam(SET_VALUE_NAME)
    public DoubleVertex getSetValue() {
        return setValue;
    }
}

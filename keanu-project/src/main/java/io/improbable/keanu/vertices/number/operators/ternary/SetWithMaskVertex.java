package io.improbable.keanu.vertices.number.operators.ternary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import org.apache.commons.lang3.NotImplementedException;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsAreScalar;

public class SetWithMaskVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends VertexImpl<TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, TensorVertex<T, TENSOR, VERTEX> {

    private final static String OPERAND_NAME = "operand";
    private final static String MASK_NAME = "mask";
    private final static String SET_VALUE_NAME = "setValue";

    private final TensorVertex<T, TENSOR, VERTEX> operand;
    private final TensorVertex<T, TENSOR, VERTEX> mask;
    private final TensorVertex<T, TENSOR, VERTEX> setValue;

    private final Class<?> type;

    @ExportVertexToPythonBindings
    public SetWithMaskVertex(@LoadVertexParam(OPERAND_NAME) TensorVertex<T, TENSOR, VERTEX> operand,
                             @LoadVertexParam(MASK_NAME) TensorVertex<T, TENSOR, VERTEX> mask,
                             @LoadVertexParam(SET_VALUE_NAME) TensorVertex<T, TENSOR, VERTEX> setValue) {
        super(checkAllShapesMatch(operand.getShape(), mask.getShape()));
        checkTensorsAreScalar("setValue must be scalar", setValue.getShape());
        this.operand = operand;
        this.mask = mask;
        this.setValue = setValue;
        this.type = operand.ofType();
        setParents(operand, mask, setValue);
    }

    @Override
    public TENSOR calculate() {
        return operand.getValue().setWithMask(mask.getValue(), setValue.getValue().scalar());
    }

    @SaveVertexParam(OPERAND_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getOperand() {
        return operand;
    }

    @SaveVertexParam(MASK_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getMask() {
        return mask;
    }

    @SaveVertexParam(SET_VALUE_NAME)
    public TensorVertex<T, TENSOR, VERTEX> getSetValue() {
        return setValue;
    }

    @Override
    public VERTEX wrap(NonProbabilisticVertex<TENSOR, VERTEX> vertex) {
        throw new NotImplementedException("Cannot wrap untyped vertex. This must be overridden by a typed vertex.");
    }

    @Override
    public Class<?> ofType() {
        return type;
    }
}

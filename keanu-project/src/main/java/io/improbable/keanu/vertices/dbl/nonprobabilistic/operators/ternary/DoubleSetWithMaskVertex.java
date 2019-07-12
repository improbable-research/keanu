package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkAllShapesMatch;
import static io.improbable.keanu.tensor.TensorShapeValidation.checkTensorsAreScalar;

public class DoubleSetWithMaskVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, NonProbabilistic<DoubleTensor> {

    private final static String OPERAND_NAME = "operand";
    private final static String MASK_NAME = "mask";
    private final static String SET_VALUE_NAME = "setValue";

    private final Vertex<DoubleTensor, ?> operand;
    private final Vertex<DoubleTensor, ?> mask;
    private final Vertex<DoubleTensor, ?> setValue;

    @ExportVertexToPythonBindings
    public DoubleSetWithMaskVertex(@LoadVertexParam(OPERAND_NAME) Vertex<DoubleTensor, ?> operand,
                                   @LoadVertexParam(MASK_NAME) Vertex<DoubleTensor, ?> mask,
                                   @LoadVertexParam(SET_VALUE_NAME) Vertex<DoubleTensor, ?> setValue) {
        super(checkAllShapesMatch(operand.getShape(), mask.getShape()));
        checkTensorsAreScalar("setValue must be scalar", setValue.getShape());
        this.operand = operand;
        this.mask = mask;
        this.setValue = setValue;
        setParents(operand, mask, setValue);
    }

    @Override
    public DoubleTensor calculate() {
        return operand.getValue().setWithMask(mask.getValue(), setValue.getValue().scalar());
    }

    @SaveVertexParam(OPERAND_NAME)
    public Vertex<DoubleTensor, ?> getOperand() {
        return operand;
    }

    @SaveVertexParam(MASK_NAME)
    public Vertex<DoubleTensor, ?> getMask() {
        return mask;
    }

    @SaveVertexParam(SET_VALUE_NAME)
    public Vertex<DoubleTensor, ?> getSetValue() {
        return setValue;
    }
}

package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

public class ConstantDoubleVertex extends VertexImpl<DoubleTensor, DoubleVertex> implements DoubleVertex, Differentiable, NonProbabilistic<DoubleTensor>, ConstantVertex {

    private final static String CONSTANT_NAME = "constant";

    @ExportVertexToPythonBindings
    public ConstantDoubleVertex(@LoadVertexParam(CONSTANT_NAME) DoubleTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantDoubleVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    public ConstantDoubleVertex(double... vector) {
        this(DoubleTensor.vector(vector));
    }

    public ConstantDoubleVertex(double[] data, long[] shape) {
        this(DoubleTensor.create(data, shape));
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        return PartialDerivative.EMPTY;
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        return Collections.emptyMap();
    }

    public DoubleTensor calculate() {
        return getValue();
    }

    @SaveVertexParam(CONSTANT_NAME)
    public DoubleTensor getConstantValue() {
        return getValue();
    }

}
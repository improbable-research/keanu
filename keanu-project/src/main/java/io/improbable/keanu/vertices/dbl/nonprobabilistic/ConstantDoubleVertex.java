package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexValue;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Collections;
import java.util.Map;

public class ConstantDoubleVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, ConstantVertex {

    @ExportVertexToPythonBindings
    public ConstantDoubleVertex(@LoadVertexValue DoubleTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantDoubleVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    public ConstantDoubleVertex(double[] vector) {
        this(DoubleTensor.create(vector));
    }

    @Override
    public PartialDerivatives forwardModeAutoDifferentiation(Map<Vertex, PartialDerivatives> derivativeOfParentsWithRespectToInputs) {
        return PartialDerivatives.OF_CONSTANT;
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {
        return Collections.emptyMap();
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return getValue();
    }

    public DoubleTensor calculate() {
        return getValue();
    }

    @Override
    public void save(NetworkSaver netSaver) {
        netSaver.save(this);
    }
}

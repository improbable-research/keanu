package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.network.NetworkWriter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ConstantDoubleVertex extends DoubleVertex implements Differentiable, NonProbabilistic<DoubleTensor>, ConstantVertex {

    @ExportVertexToPythonBindings
    public ConstantDoubleVertex(DoubleTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantDoubleVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    public ConstantDoubleVertex(double[] vector) {
        this(DoubleTensor.create(vector));
    }

    public ConstantDoubleVertex(Map<String, Vertex> parentMap, KeanuSavedBayesNet.VertexValue initialValue) {
        this(parseValue(initialValue));
    }

    private static DoubleTensor parseValue(KeanuSavedBayesNet.VertexValue value) {
        if (value.getValueTypeCase() != KeanuSavedBayesNet.VertexValue.ValueTypeCase.DOUBLEVAL) {
            throw new IllegalArgumentException("Non Double Value specified for Double Vertex");
        } else {
            return DoubleTensor.create(
                Doubles.toArray(value.getDoubleVal().getValuesList()),
                Longs.toArray(value.getDoubleVal().getShapeList())
            );
        }
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

    @Override
    public Map<String, Vertex> getParentsMap() {
        return new HashMap<>();
    }

    public DoubleTensor calculate() {
        return getValue();
    }

    @Override
    public void save(NetworkWriter netWriter) {
        netWriter.save(this);
    }
}

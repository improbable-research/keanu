package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexValue;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantBoolVertex extends BoolVertex implements ConstantVertex, NonProbabilistic<BooleanTensor> {

    public static final BoolVertex TRUE = new ConstantBoolVertex(true);
    public static final BoolVertex FALSE = new ConstantBoolVertex(false);

    @ExportVertexToPythonBindings
    public ConstantBoolVertex(@LoadVertexValue BooleanTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantBoolVertex(boolean constant) {
        this(BooleanTensor.scalar(constant));
    }

    public ConstantBoolVertex(boolean[] vector) {
        this(BooleanTensor.create(vector));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public BooleanTensor calculate() {
        return getValue();
    }

    @Override
    public void save(NetworkSaver netSaver) {
        netSaver.save(this);
    }
}

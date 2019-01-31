package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public class ConstantBooleanVertex extends BooleanVertex implements ConstantVertex, NonProbabilistic<BooleanTensor> {

    public static final BooleanVertex TRUE = new ConstantBooleanVertex(true);
    public static final BooleanVertex FALSE = new ConstantBooleanVertex(false);
    private final static String CONSTANT_NAME = "constant";

    @ExportVertexToPythonBindings
    public ConstantBooleanVertex(@LoadVertexParam(CONSTANT_NAME) BooleanTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantBooleanVertex(boolean constant) {
        this(BooleanTensor.scalar(constant));
    }

    public ConstantBooleanVertex(boolean[] vector) {
        this(BooleanTensor.create(vector));
    }

    public ConstantBooleanVertex(boolean[] data, long[] shape) {
        this(BooleanTensor.create(data, shape));
    }

    @Override
    public BooleanTensor calculate() {
        return getValue();
    }

    @Override
    public void save(NetworkSaver netSaver) {
        netSaver.save(this);
    }

    @SaveVertexParam(CONSTANT_NAME)
    public BooleanTensor getConstantValue() {
        return getValue();
    }
}

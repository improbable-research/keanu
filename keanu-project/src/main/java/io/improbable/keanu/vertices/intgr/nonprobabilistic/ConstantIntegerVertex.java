package io.improbable.keanu.vertices.intgr.nonprobabilistic;


import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class ConstantIntegerVertex extends IntegerVertex implements ConstantVertex, NonProbabilistic<IntegerTensor> {

    private final static String CONSTANT_NAME = "constant";

    @ExportVertexToPythonBindings
    public ConstantIntegerVertex(@LoadVertexParam(CONSTANT_NAME) IntegerTensor constant) {
        super(constant.getShape());
        setValue(constant);
    }

    public ConstantIntegerVertex(int[] vector) {
        this(IntegerTensor.create(vector));
    }

    public ConstantIntegerVertex(int constant) {
        this(IntegerTensor.scalar(constant));
    }

    public ConstantIntegerVertex(int[] data, long[] shape) {
        this(IntegerTensor.create(data, shape));
    }

    @Override
    public void save(NetworkSaver netSaver) {
        netSaver.save(this);
    }

    @Override
    public IntegerTensor calculate() {
        return getValue();
    }

    @SaveVertexParam(CONSTANT_NAME)
    public IntegerTensor getConstantValue() {
        return getValue();
    }
}

package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class ConstantVertex {

    private ConstantVertex() {
    }

    public static ConstantBoolVertex of(boolean value) {
        return new ConstantBoolVertex(value);
    }

    public static ConstantBoolVertex of(boolean[] value) {
        return new ConstantBoolVertex(value);
    }

    public static ConstantBoolVertex of(BooleanTensor value) {
        return new ConstantBoolVertex(value);
    }

    public static ConstantIntegerVertex of(int value) {
        return new ConstantIntegerVertex(value);
    }

    public static ConstantIntegerVertex of(int[] value) {
        return new ConstantIntegerVertex(value);
    }

    public static ConstantIntegerVertex of(IntegerTensor value) {
        return new ConstantIntegerVertex(value);
    }

    public static ConstantDoubleVertex of(double value) {
        return new ConstantDoubleVertex(value);
    }

    public static ConstantDoubleVertex of(double[] value) {
        return new ConstantDoubleVertex(value);
    }

    public static ConstantDoubleVertex of(DoubleTensor value) {
        return new ConstantDoubleVertex(value);
    }

    public static <GENERIC> ConstantGenericVertex<GenericTensor<GENERIC>> of(GENERIC value) {
        return new ConstantGenericVertex<>(new GenericTensor<>(value));
    }

    public static <GENERIC> ConstantGenericVertex<GenericTensor<GENERIC>> of(GENERIC[] values) {
        return new ConstantGenericVertex<>(new GenericTensor<>(values, new int[]{1, values.length}));
    }

    public static <TENSOR extends Tensor> ConstantGenericVertex<TENSOR> of(TENSOR tensor) {
        return new ConstantGenericVertex<>(tensor);
    }
}

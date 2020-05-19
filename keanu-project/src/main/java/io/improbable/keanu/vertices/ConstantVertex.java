package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.ConstantGenericTensorVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;

public interface ConstantVertex {

    static <TYPED extends ConstantVertex> TYPED scalar(Object obj) {
        if (obj.getClass().isAssignableFrom(Double.class)) {
            return (TYPED) of((Double) obj);
        } else if (obj.getClass().isAssignableFrom(Integer.class)) {
            return (TYPED) of((Integer) obj);
        } else if (obj.getClass().isAssignableFrom(Boolean.class)) {
            return (TYPED) of((Boolean) obj);
        } else {
            return (TYPED) of(obj);
        }
    }

    static ConstantBooleanVertex of(Boolean value) {
        return new ConstantBooleanVertex(value);
    }

    static ConstantBooleanVertex of(boolean value) {
        return new ConstantBooleanVertex(value);
    }

    static ConstantBooleanVertex of(boolean... value) {
        return new ConstantBooleanVertex(value);
    }

    static ConstantBooleanVertex of(boolean[] value, long... shape) {
        return new ConstantBooleanVertex(value, shape);
    }

    static ConstantBooleanVertex of(BooleanTensor value) {
        return new ConstantBooleanVertex(value);
    }

    static ConstantIntegerVertex of(Integer value) {
        return new ConstantIntegerVertex(value);
    }

    static ConstantIntegerVertex of(int value) {
        return new ConstantIntegerVertex(value);
    }

    static ConstantIntegerVertex of(int... value) {
        return new ConstantIntegerVertex(value);
    }

    static ConstantIntegerVertex of(int[] value, long... shape) {
        return new ConstantIntegerVertex(value, shape);
    }

    static ConstantIntegerVertex of(IntegerTensor value) {
        return new ConstantIntegerVertex(value);
    }

    static ConstantDoubleVertex of(Double value) {
        return new ConstantDoubleVertex(value);
    }

    static ConstantDoubleVertex of(double value) {
        return new ConstantDoubleVertex(value);
    }

    static ConstantDoubleVertex of(double... value) {
        return new ConstantDoubleVertex(value);
    }

    static ConstantDoubleVertex of(double[] value, long... shape) {
        return new ConstantDoubleVertex(value, shape);
    }

    static ConstantDoubleVertex of(DoubleTensor value) {
        return new ConstantDoubleVertex(value);
    }

    static <GENERIC> ConstantGenericTensorVertex<GENERIC> of(GENERIC value) {
        return new ConstantGenericTensorVertex<>(GenericTensor.scalar(value));
    }

    static <GENERIC> ConstantGenericTensorVertex<GENERIC> of(GENERIC[] values) {
        return new ConstantGenericTensorVertex<>(GenericTensor.create(values));
    }
}

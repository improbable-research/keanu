package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.factory.Nd4j;

public class TypedINDArrayFactory {

    private static final DataType DEFAULT_FLOATING_POINT_TYPE = DataType.DOUBLE;

    public static INDArray create(double[] data, long[] shape) {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.createFromArray(data).reshape(shape);
    }

    public static INDArray create(int[] data, long[] shape) {
        Nd4j.setDefaultDataTypes(DataType.INT, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.createFromArray(data).reshape(shape);
    }

    public static INDArray ones(long[] shape, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.ones(shape);
    }

    public static INDArray eye(long n, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, DEFAULT_FLOATING_POINT_TYPE);
        if (n == 0) {
            return Nd4j.scalar(1.0);
        }
        return Nd4j.eye(n);
    }

    public static INDArray zeros(long[] shape, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.zeros(shape);
    }

    public static INDArray linspace(double start, double end, int numberOfPoints, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.getExecutioner().exec(
            new Linspace(Nd4j.createUninitialized(new long[]{numberOfPoints}, Nd4j.order()), start, end)
        );
    }

    public static INDArray arange(double start, double end) {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.arange(start, end);
    }

    public static INDArray arange(int start, int end) {
        Nd4j.setDefaultDataTypes(DataType.INT, DEFAULT_FLOATING_POINT_TYPE);
        return Nd4j.arange(start, end);
    }

}

package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.factory.Nd4j;

public class TypedINDArrayFactory {

    public static INDArray create(double[] data, long[] shape, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        switch (shape.length) {
            case 0:
                Preconditions.checkArgument(data.length == 1, "Scalar shape must have only one data element.");
                return scalar(data[0], bufferType);
            case 1:
                return vector(data, bufferType);
            default:
                DataBuffer buffer = Nd4j.getDataBufferFactory().createDouble(data);
                return Nd4j.create(buffer, shape);
        }
    }

    public static INDArray valueArrayOf(long[] shape, double value, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        switch (shape.length) {
            case 0:
                return scalar(value, bufferType);
            case 1:
                return reshapeToVector(Nd4j.valueArrayOf(shape, value));
            default:
                return Nd4j.valueArrayOf(shape, value);
        }
    }

    public static INDArray scalar(double scalarValue, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        return Nd4j.trueScalar(scalarValue);
    }

    public static INDArray vector(double[] vectorValues, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        return Nd4j.trueVector(vectorValues);
    }

    public static INDArray ones(long[] shape, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        switch (shape.length) {
            case 0:
                return scalar(1.0, bufferType);
            case 1:
                return reshapeToVector(Nd4j.ones(shape));
            default:
                return Nd4j.ones(shape);
        }
    }

    public static INDArray eye(long n, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        if (n == 0) {
            return scalar(1.0, bufferType);
        }
        return Nd4j.eye(n);
    }

    public static INDArray zeros(long[] shape, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        return Nd4j.zeros(shape);
    }

    public static INDArray linspace(double start, double end, int numberOfPoints, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        return Nd4j.getExecutioner().exec(
            new Linspace(Nd4j.createUninitialized(new long[]{numberOfPoints}, Nd4j.order()), start, end)
        );
    }

    public static INDArray arange(double start, double end, DataType bufferType) {
        Nd4j.setDefaultDataTypes(bufferType, bufferType);
        return reshapeToVector(Nd4j.arange(start, end));
    }

    private static INDArray reshapeToVector(INDArray vector) {
        if (vector.shape().length > 1) {
            vector.setShapeAndStride(new int[]{(int) vector.shape()[1]}, new int[]{1});
        }
        return vector;
    }
}

package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TypedINDArrayFactory {

    public static INDArray create(double[] data, long[] shape, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        DataBuffer buffer = Nd4j.getDataBufferFactory().createDouble(data);
        return Nd4j.create(buffer, shape);
    }

    public static INDArray create(int[] data, long[] shape, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        DataBuffer buffer = Nd4j.getDataBufferFactory().createDouble(data);
        return Nd4j.create(buffer, shape);
    }

    public static INDArray valueArrayOf(long[] shape, double value, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.valueArrayOf(shape, value);
    }

    public static INDArray scalar(double scalarValue, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.scalar(scalarValue);
    }

    public static INDArray ones(long[] shape, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.ones(shape);
    }

    public static INDArray eye(int n, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.eye(n);
    }

    public static INDArray zeros(long[] shape, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.zeros(shape);
    }

    public static INDArray linspace(double start, double end, int numberOfPoints, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.linspace(start, end, numberOfPoints);
    }

    public static INDArray arange(double start, double end, DataBuffer.Type bufferType) {
        Nd4j.setDataType(bufferType);
        return Nd4j.arange(start, end);
    }
}

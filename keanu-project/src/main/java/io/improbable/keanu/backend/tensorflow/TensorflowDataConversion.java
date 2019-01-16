package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import lombok.experimental.UtilityClass;
import org.tensorflow.Tensor;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

@UtilityClass
class TensorflowDataConversion {

    static DoubleTensor toDoubleTensor(Tensor<?> tensor) {

        DoubleBuffer buffer = DoubleBuffer.allocate(tensor.numElements());
        tensor.writeTo(buffer);
        double[] resultAsArray = buffer.array();

        long[] shape = tensor.shape();
        if (shape.length == 0) {
            return DoubleTensor.scalar(resultAsArray[0]);
        } else if (shape.length == 1) {
            return DoubleTensor.create(resultAsArray);
        } else {
            return DoubleTensor.create(resultAsArray, shape);
        }
    }

    static BooleanTensor toBooleanTensor(Tensor<?> tensor) {

        ByteBuffer buffer = ByteBuffer.allocate(tensor.numElements());
        tensor.writeTo(buffer);
        boolean[] resultAsArray = byte2bool(buffer.array());

        long[] shape = tensor.shape();
        if (shape.length == 0) {
            return BooleanTensor.scalar(resultAsArray[0]);
        } else if (shape.length == 1) {
            return BooleanTensor.create(resultAsArray);
        } else {
            return BooleanTensor.create(resultAsArray, shape);
        }
    }

    private static boolean[] byte2bool(byte[] array) {
        boolean[] out = new boolean[array.length];
        for (int i = 0; i < array.length; i++) {
            out[i] = array[i] != 0;
        }
        return out;
    }

    static IntegerTensor toIntegerTensor(Tensor<?> tensor) {

        IntBuffer buffer = IntBuffer.allocate(tensor.numElements());
        tensor.writeTo(buffer);
        int[] resultAsArray = buffer.array();

        long[] shape = tensor.shape();
        if (shape.length == 0) {
            return IntegerTensor.scalar(resultAsArray[0]);
        } else if (shape.length == 1) {
            return IntegerTensor.create(resultAsArray);
        } else {
            return IntegerTensor.create(resultAsArray, shape);
        }
    }

    static Tensor<Double> toTensorFlow(DoubleTensor keanuTensor) {
        return toTensorFlow(keanuTensor.getShape(), keanuTensor.asFlatDoubleArray());
    }

    static Tensor<Boolean> toTensorFlow(BooleanTensor keanuTensor) {
        return toTensorFlow(keanuTensor.getShape(), keanuTensor.asFlatArray());
    }

    static Tensor<Integer> toTensorFlow(IntegerTensor keanuTensor) {
        return toTensorFlow(keanuTensor.getShape(), keanuTensor.asFlatIntegerArray());
    }

    static Tensor<Double> toTensorFlow(long[] shape, double[] data) {
        return Tensor.create(
            shape,
            DoubleBuffer.wrap(data)
        );
    }

    static Tensor<Boolean> toTensorFlow(long[] shape, Boolean[] data) {
        return Tensor.create(
            Boolean.class,
            shape,
            ByteBuffer.wrap(bool2byte(data))
        );
    }

    static Tensor<Integer> toTensorFlow(long[] shape, int[] data) {
        return Tensor.create(
            shape,
            IntBuffer.wrap(data)
        );
    }

    private static byte[] bool2byte(Boolean[] array) {
        byte[] out = new byte[array.length];
        for (int i = 0; i < array.length; i++) {
            out[i] = array[i] ? (byte) 1 : (byte) 0;
        }
        return out;
    }
}

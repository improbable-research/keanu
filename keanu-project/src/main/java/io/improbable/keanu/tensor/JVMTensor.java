package io.improbable.keanu.tensor;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.buffer.JVMBuffer;

public class JVMTensor {

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
    JVMTensorBroadcast.ResultWrapper<T, B> diag(int rank, long[] shape,
                                                B buffer, JVMBuffer.ArrayWrapperFactory<T, B> factory) {

        B newBuffer;
        long[] newShape;
        if (rank == 1) {
            int n = buffer.getLength();
            newBuffer = factory.createNew(Ints.checkedCast((long) n * (long) n));
            ;
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i), i * n + i);
            }
            newShape = new long[]{n, n};
        } else if (rank == 2 && shape[0] == shape[1]) {
            int n = Ints.checkedCast(shape[0]);
            newBuffer = factory.createNew(n);
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i * n + i), i);
            }
            newShape = new long[]{n};
        } else {
            throw new IllegalArgumentException("Diag is only valid for vectors or square matrices");
        }

        return new JVMTensorBroadcast.ResultWrapper<>(newBuffer, newShape, TensorShape.getRowFirstStride(newShape));
    }
}

package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import lombok.AllArgsConstructor;

@AllArgsConstructor
public class ResultWrapper<T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> {
    public final B outputBuffer;
    public final long[] outputShape;
    public final long[] outputStride;
}

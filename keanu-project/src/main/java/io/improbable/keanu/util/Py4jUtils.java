package io.improbable.keanu.util;

import lombok.experimental.UtilityClass;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@UtilityClass
public class Py4jUtils {
    private static final int JAVA_DOUBLE_SIZE = 8;
    private static final int JAVA_INT_SIZE = 4;

    public byte[] toByteArray(double[] doubleArray) {
        ByteBuffer doubleBuffer = ByteBuffer.allocate(JAVA_DOUBLE_SIZE * doubleArray.length);
        // Java defaults to BIG_ENDIAN. LITTLE_ENDIAN is what is expected by numpy to construct an ndarray.
        doubleBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (double element : doubleArray) {
            doubleBuffer.putDouble(element);
        }
        byte[] byteArray = doubleBuffer.array();
        return byteArray;
    }

    public byte[] toByteArray(int[] integerArray) {
        ByteBuffer integerBuffer = ByteBuffer.allocate(JAVA_INT_SIZE * integerArray.length);
        // Java defaults to BIG_ENDIAN. LITTLE_ENDIAN is what is expected by numpy to construct an ndarray.
        integerBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int element : integerArray) {
            integerBuffer.putInt(element);
        }
        byte[] byteArray = integerBuffer.array();
        return byteArray;
    }

    public byte[] toByteArray(Boolean[] booleanArray) {
        ByteBuffer booleanBuffer = ByteBuffer.allocate(booleanArray.length);
        booleanBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (boolean element : booleanArray) {
            byte booleanByte = element ? (byte) 0x01 : 0x00;
            booleanBuffer.put(booleanByte);
        }
        byte[] byteArray = booleanBuffer.array();
        return byteArray;
    }
}

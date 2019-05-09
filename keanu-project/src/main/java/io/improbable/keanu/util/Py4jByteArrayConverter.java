package io.improbable.keanu.util;

import lombok.experimental.UtilityClass;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.BitSet;

@UtilityClass
public class Py4jByteArrayConverter {

    public byte[] toByteArray(double[] doubleArray) {
        ByteBuffer doubleBuffer = ByteBuffer.allocate(Double.SIZE / Byte.SIZE * doubleArray.length);
        // Java defaults to BIG_ENDIAN. LITTLE_ENDIAN is what is expected by numpy to construct an ndarray.
        doubleBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (double element : doubleArray) {
            doubleBuffer.putDouble(element);
        }
        byte[] byteArray = doubleBuffer.array();
        return byteArray;
    }

    public byte[] toByteArray(int[] integerArray) {
        ByteBuffer integerBuffer = ByteBuffer.allocate(Integer.SIZE / Byte.SIZE * integerArray.length);
        // Java defaults to BIG_ENDIAN. LITTLE_ENDIAN is what is expected by numpy to construct an ndarray.
        integerBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int element : integerArray) {
            integerBuffer.putInt(element);
        }
        byte[] byteArray = integerBuffer.array();
        return byteArray;
    }

    public byte[] toByteArray(boolean[] booleanArray) {
        ByteBuffer booleanBuffer = ByteBuffer.allocate(booleanArray.length);
        booleanBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (boolean element : booleanArray) {
            byte booleanByte = element ? (byte) 0x01 : 0x00;
            booleanBuffer.put(booleanByte);
        }
        byte[] byteArray = booleanBuffer.array();
        return byteArray;
    }

    public static double[] toDoubleArray(byte[] byteArray, int itemSizeBytes){
        double[] doubles = new double[byteArray.length / itemSizeBytes];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ByteBuffer.wrap(byteArray, i*itemSizeBytes, itemSizeBytes).order(ByteOrder.LITTLE_ENDIAN).getDouble();
        }
        return doubles;
    }

    public static int[] toIntegerArray(byte[] byteArray, int itemSizeBytes){
        int[] ints = new int[byteArray.length / itemSizeBytes];
        for(int i = 0; i < ints.length; i++) {
            ints[i] = ByteBuffer.wrap(byteArray, i*itemSizeBytes, itemSizeBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
        }
        return ints;
    }

    public static boolean[] toBooleanArray(byte[] byteArray, int numberOfBooleansInArray) {
        BitSet bits = BitSet.valueOf(byteArray);
        boolean[] bools = new boolean[numberOfBooleansInArray];
        int numberOfBitsToIterateThrough = ((int) Math.ceil((numberOfBooleansInArray - 1.0) / 8.0) + 1) * 8;
        for (int i = bits.nextSetBit(0); i != -1 && i < numberOfBitsToIterateThrough; i = bits.nextSetBit(i+1)) {
            int position = getBigEndianPosition(i);
            if (position < numberOfBooleansInArray) {
                bools[position] = true;
            }
        }
        return bools;
    }

    private int getBigEndianPosition(int index){
        int byteNumber = index / 8;
        int bitNumber = index % 8;
        return byteNumber * 8 + (7 - bitNumber);
    }
}

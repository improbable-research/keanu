package io.improbable.keanu.tensor;

import java.nio.ByteBuffer;

public class ByteArrayConverter {
    public static double[] toDoubleArray(byte[] byteArray){
        int times = Double.SIZE / Byte.SIZE;
        double[] doubles = new double[byteArray.length / times];
        for(int i=0;i<doubles.length;i++){
            doubles[i] = ByteBuffer.wrap(byteArray, i*times, times).getDouble();
        }
        return doubles;
    }

    public static int[] toIntegerArray(byte[] byteArray){
        int times = Integer.SIZE / Byte.SIZE;
        int[] ints = new int[byteArray.length / times];
        for(int i=0;i<ints.length;i++){
            ints[i] = ByteBuffer.wrap(byteArray, i*times, times).getInt();
        }
        return ints;
    }
}

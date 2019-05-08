package io.improbable.keanu.util;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class Py4jByteArrayConverterTest {

    @Test
    public void canCreateIntegerArrayFromByteArray() {
        byte byte1 = Byte.parseByte("00000110", 2);
        byte byte2 = Byte.parseByte("00000011", 2);
        byte zeroByte = Byte.parseByte("00000000", 2);
        byte[] bytes = new byte[] {
            byte1, zeroByte, zeroByte, zeroByte,
            byte2, zeroByte, zeroByte, zeroByte
        };

        int[] ints = Py4jByteArrayConverter.toIntegerArray(bytes, 4);

        assertThat(ints.length, is(2));
        assertThat(ints[0], is(6));
        assertThat(ints[1], is(3));
    }

    @Test
    public void canCreateDoubleArrayFromByteArray() {
        byte byte1 = Byte.parseByte("01000000", 2);
        byte byte2 = Byte.parseByte("00011000", 2);
        byte byte3 = Byte.parseByte("00001000", 2);
        byte zeroByte = Byte.parseByte("00000000", 2);
        byte[] bytes = new byte[] {
            zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, byte2, byte1,
            zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, byte3, byte1
        };

        double[] doubles = Py4jByteArrayConverter.toDoubleArray(bytes, 8);

        assertThat(doubles.length, is(2));
        assertThat(doubles[0], is(6.0));
        assertThat(doubles[1], is(3.0));
    }
}

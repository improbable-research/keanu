package io.improbable.keanu.tensor;

import org.junit.Test;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

public class ByteArrayConverterTest {

    @Test
    public void canCreateIntegerArrayFromByteArray() {
        byte byte1 = Byte.parseByte("00000110", 2);
        byte byte2 = Byte.parseByte("00000011", 2);
        byte zeroByte = Byte.parseByte("00000000", 2);
        byte[] bytes = new byte[] {
            zeroByte, zeroByte, zeroByte, byte1,
            zeroByte, zeroByte, zeroByte, byte2
        };

        int[] ints = ByteArrayConverter.toIntegerArray(bytes);

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
            byte1, byte2, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte,
            byte1, byte3, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte, zeroByte,
        };

        double[] doubles = ByteArrayConverter.toDoubleArray(bytes);

        assertThat(doubles.length, is(2));
        assertThat(doubles[0], is(6.0));
        assertThat(doubles[1], is(3.0));
    }


}

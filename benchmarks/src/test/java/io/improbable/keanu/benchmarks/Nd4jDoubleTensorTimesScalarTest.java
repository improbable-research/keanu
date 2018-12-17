package io.improbable.keanu.benchmarks;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class Nd4jDoubleTensorTimesScalarTest {

    @Test
    public void theScalarsHaveTheExpectedRank() {
        assertArrayEquals(Nd4jDoubleTensorTimesScalar.scalars[0].getShape(), new long[] {});
        assertArrayEquals(Nd4jDoubleTensorTimesScalar.scalars[1].getShape(), new long[] {1});
        assertArrayEquals(Nd4jDoubleTensorTimesScalar.scalars[2].getShape(), new long[] {1, 1});
        assertArrayEquals(Nd4jDoubleTensorTimesScalar.scalars[3].getShape(), new long[] {});
        assertEquals(Nd4jDoubleTensorTimesScalar.scalars.length, 4);
    }
}
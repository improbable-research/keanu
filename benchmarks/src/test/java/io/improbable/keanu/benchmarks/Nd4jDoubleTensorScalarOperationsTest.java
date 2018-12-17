package io.improbable.keanu.benchmarks;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class Nd4jDoubleTensorScalarOperationsTest {

    @Test
    public void theScalarsHaveTheExpectedRank() {
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.scalars[0].getShape(), new long[] {});
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.scalars[1].getShape(), new long[] {1});
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.scalars[2].getShape(), new long[] {1, 1});
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.scalars[3].getShape(), new long[] {});
        assertEquals(Nd4jDoubleTensorScalarOperations.scalars.length, 4);
    }
}
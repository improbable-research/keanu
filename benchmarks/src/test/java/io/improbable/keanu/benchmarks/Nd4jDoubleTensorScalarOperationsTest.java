package io.improbable.keanu.benchmarks;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class Nd4jDoubleTensorScalarOperationsTest {

    @Test
    public void theScalarsHaveTheExpectedRank() {
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.RANK_0_SCALAR_TENSOR.getShape(), new long[] {});
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.RANK_1_SCALAR_TENSOR.getShape(), new long[] {1});
        assertArrayEquals(Nd4jDoubleTensorScalarOperations.RANK_2_SCALAR_TENSOR.getShape(), new long[] {1, 1});
    }
}
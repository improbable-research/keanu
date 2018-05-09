package io.improbable.keanu.vertices.dbltensor;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;

public class Nd4jDoubleTensorTest {

    @Test
    public void canElementWiseMultiplyMatrix() {
        Nd4jDoubleTensor a = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        Nd4jDoubleTensor b = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});

        DoubleTensor result = a.times(b);

        assertTrue(Arrays.equals(result.getLinearView(), new double[]{1, 4, 9, 16}));
    }

    @Test
    public void canMultiplyMatrixByScalar() {
        Nd4jDoubleTensor a = Nd4jDoubleTensor.create(new double[]{1, 2, 3, 4}, new int[]{2, 2});
        Nd4jDoubleTensor b = Nd4jDoubleTensor.scalar(2.0);

        DoubleTensor result = a.times(b);

        assertTrue(Arrays.equals(result.getLinearView(), new double[]{2, 4, 6, 8}));
    }

}

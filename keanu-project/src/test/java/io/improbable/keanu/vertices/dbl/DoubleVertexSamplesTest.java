package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;

public class DoubleVertexSamplesTest {

    @Test
    public void doesCalculateAverage() {

        DoubleTensor a = DoubleTensor.create(new double[]{0, 16, 4}, new int[]{1, 3});
        DoubleTensor b = DoubleTensor.create(new double[]{-4, -8, 4}, new int[]{1, 3});
        DoubleTensor c = DoubleTensor.create(new double[]{8, -4, 12}, new int[]{1, 3});
        DoubleTensor d = DoubleTensor.create(new double[]{4, 4, 8}, new int[]{1, 3});

        DoubleVertexSamples samples = new DoubleVertexSamples(Arrays.asList(a, b, c, d));

        DoubleTensor averages = samples.getAverages();

        assertArrayEquals(new double[]{2.0, 2.0, 7.0}, averages.asFlatDoubleArray(), 0.0);

    }
}

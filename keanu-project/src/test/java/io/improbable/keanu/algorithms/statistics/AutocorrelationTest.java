package io.improbable.keanu.algorithms.statistics;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;

import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

public class AutocorrelationTest {

    @Test
    public void autocorrAtLagZeroIsOne() {
        double[] samples = generateUniformRandomArray(20);
        double[] autocorr = Autocorrelation.calculate(samples);
        assertEquals(autocorr[0], 1.0, 0.0);
    }

    private double[] generateUniformRandomArray(int length) {
        ContinuousDistribution uniform = Uniform.withParameters(DoubleTensor.ZERO_SCALAR, DoubleTensor.scalar(1000));
        return uniform.sample(new long[]{length}, KeanuRandom.getDefaultRandom()).asFlatDoubleArray();
    }

    @Test
    public void randomlyGeneratedSamplesHaveCloseToZeroAutocorrelationAtLowLags() {
        double[] samples = generateUniformRandomArray(500);
        double[] autocorr = Autocorrelation.calculate(samples);

        for (int lag = 1; lag < 10; lag++) {
            assertThat(autocorr[lag], lessThan(0.1));
        }
    }

    /*
      import statsmodels.api as sm
      import numpy as np
      x = np.array([1,2,3,4,5,6,7,8])
      print(sm.tsa.stattools.acf(x))
    */
    @Test
    public void increasingSequenceReturnsCorrectAutocorr() {
        double[] samples = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] autocorr = Autocorrelation.calculate(samples);
        double[] expected = {1., 0.625, 0.27380952, -0.0297619, -0.26190476,
            -0.39880952, -0.41666667, -0.29166667};
        assertArrayEquals(expected, autocorr, 0.0001);

    }

}

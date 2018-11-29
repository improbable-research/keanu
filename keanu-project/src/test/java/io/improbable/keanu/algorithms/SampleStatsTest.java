package io.improbable.keanu.algorithms;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.Uniform;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;

import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThat;

public class SampleStatsTest {
    @Test
    public void randomlyGeneratedSamplesHaveCloseToZeroAutocorrelationAtLowLags() {
        ContinuousDistribution uniform = Uniform.withParameters(DoubleTensor.ZERO_SCALAR, DoubleTensor.scalar(1000));
        uniform.sample(new long[]{500}, KeanuRandom.getDefaultRandom());
        double samples[] = new double[500];
        for (int i = 0; i < 500; i++) {
            samples[i] = uniform.sample(new long[]{1}, KeanuRandom.getDefaultRandom()).scalar();
        }
        double[] autocorr = SampleStats.acf(samples);

        assertThat(autocorr[80], lessThan(0.1));
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
        double[] autocorr = SampleStats.acf(samples);
        double[] expected = {1., 0.625, 0.27380952, -0.0297619, -0.26190476,
            -0.39880952, -0.41666667, -0.29166667};
        assertArrayEquals(expected, autocorr, 0.0001);

    }

}

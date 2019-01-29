package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.math3.random.MersenneTwister;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class KeanuRandomTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void canSampleGaussianScalar() {
        assertEquals(0, random.nextGaussian(new long[0]).getRank());
        assertEquals(1, random.nextGaussian(new long[]{1}).getRank());
        assertEquals(2, random.nextGaussian(new long[]{1, 1}).getRank());
        assertEquals(2, random.nextGaussian(new long[]{2, 3}).getRank());
    }

    @Test
    public void canSampleUniformScalar() {
        assertEquals(0, random.nextDouble(new long[0]).getRank());
        assertEquals(1, random.nextDouble(new long[]{1}).getRank());
        assertEquals(2, random.nextDouble(new long[]{1, 1}).getRank());
        assertEquals(2, random.nextDouble(new long[]{2, 3}).getRank());
    }

    @Test
    public void canSampleGammaScalar() {
        assertEquals(0, random.nextGamma(new long[0], DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(1, random.nextGamma(new long[]{1}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(2, random.nextGamma(new long[]{1, 1}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(2, random.nextGamma(new long[]{2, 3}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
    }

    @Test
    public void canSampleLaplaceScalar() {
        assertEquals(0, random.nextLaplace(new long[0], DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(1, random.nextLaplace(new long[]{1}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(2, random.nextLaplace(new long[]{1, 1}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
        assertEquals(2, random.nextLaplace(new long[]{2, 3}, DoubleTensor.scalar(2), DoubleTensor.scalar(2)).getRank());
    }

    @Test
    public void randomNumberGenerationIsPlatformIndependent() {
        KeanuRandom keanuRandom = new KeanuRandom(1);
        assertEquals(keanuRandom.nextDouble(), new MersenneTwister(1L).nextDouble());
        assertEquals(keanuRandom.nextDouble(), 0.41782887182714457, 1e-16);
    }
}

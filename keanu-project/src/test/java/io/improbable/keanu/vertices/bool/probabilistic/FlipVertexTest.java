package io.improbable.keanu.vertices.bool.probabilistic;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class FlipVertexTest {

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        Flip flip = new Flip(expectedShape, 0.25);
        BooleanTensor samples = flip.sample(new KeanuRandom(1));
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        Flip flip = new Flip(new int[]{1, 2}, probTrue);
        double actualLogPmf = flip.logProb(BooleanTensor.create(new boolean[]{true, false}));
        double expectedLogPmf = Math.log(probTrue) + Math.log(1 - probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

}

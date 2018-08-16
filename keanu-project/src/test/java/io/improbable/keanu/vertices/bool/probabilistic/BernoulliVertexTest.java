package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class BernoulliVertexTest {

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        BernoulliVertex bernoulliVertex = new BernoulliVertex(expectedShape, 0.25);
        BooleanTensor samples = bernoulliVertex.sample(new KeanuRandom(1));
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        BernoulliVertex bernoulliVertex = new BernoulliVertex(new int[]{1, 2}, probTrue);
        double actualLogPmf = bernoulliVertex.logPmf(BooleanTensor.create(new boolean[]{true, false}));
        double expectedLogPmf = Math.log(probTrue) + Math.log(1 - probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

}

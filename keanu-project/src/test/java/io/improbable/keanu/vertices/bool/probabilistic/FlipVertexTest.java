package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class FlipVertexTest {

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        Flip flip = new Flip(expectedShape, 0.25);
        BooleanTensor samples = flip.sample(new KeanuRandom(1));
        assertArrayEquals(expectedShape, samples.getShape());
    }
}

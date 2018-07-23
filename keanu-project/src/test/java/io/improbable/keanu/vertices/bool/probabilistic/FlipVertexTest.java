package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FlipVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        Flip flip = new Flip(expectedShape, 0.25);
        BooleanTensor samples = flip.sample();
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        Flip flip = new Flip(new int[]{1, 2}, probTrue);
        double actualLogPmf = flip.logPmf(BooleanTensor.create(new boolean[]{true, true}));
        double expectedLogPmf = Math.log(probTrue) + Math.log(probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParam() {

        DoubleVertex A = new GaussianVertex(new int[]{1, 2}, 0, 1);
        A.setValue(new double[]{0.25, 0.6});
        DoubleVertex B = new GaussianVertex(new int[]{1, 2}, 0, 1);
        B.setValue(new double[]{0.5, 0.2});
        DoubleVertex C = A.times(B);
        Flip D = new Flip(C);

        Map<Long, DoubleTensor> dLogPmf = D.dLogPmf(BooleanTensor.create(new boolean[]{true, false}));

        DoubleTensor expectedWrtA = DoubleTensor.create(new double[]{0.5, -0.2});
        DoubleTensor expectedWrtB = DoubleTensor.create(new double[]{0.25, -0.6});

        assertEquals(expectedWrtA, dLogPmf.get(A.getId()));
        assertEquals(expectedWrtB, dLogPmf.get(B.getId()));

    }
}

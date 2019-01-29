package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.junit.Rule;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class GeometricVertexTest {

    @Rule
    public DeterministicRule myRule = new DeterministicRule();

    @Test
    public void logPdfIsCorrect() {
        double p = 0.25;
        GeometricVertex myVertex = new GeometricVertex(p);

        for (int i = 1; i < 10; i++) {
            assertEquals(getExpectedPmf(p, i), myVertex.logProb(IntegerTensor.create(i)), 1e-6);
        }
    }

    private double getExpectedPmf(double p, int n) {
        return Math.log(Math.pow(1 - p, n - 1) * p);
    }

    @Test
    public void canSampleFromGeometric() {
        double p = 0.5;

        GeometricVertex myVertex = new GeometricVertex(new long[] {1, 100000}, p);
        IntegerTensor samples = myVertex.sample();

    }

    private void checkSamplesMatchPMF(IntegerTensor samples) {

    }
}

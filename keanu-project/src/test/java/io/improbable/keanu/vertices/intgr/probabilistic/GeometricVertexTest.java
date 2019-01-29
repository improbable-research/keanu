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
    public void logProbIsCorrectScalar() {
        double p = 0.25;
        GeometricVertex myVertex = new GeometricVertex(p);

        for (int i = 1; i < 20; i++) {
            assertEquals(getExpectedPmf(p, i), myVertex.logProb(IntegerTensor.create(i)), 1e-6);
        }
    }

    @Test
    public void logProbIsCorrectVector() {
        double p = 0.8;
        int[] values = new int[] {3, 5, 6};
        GeometricVertex myVertex = new GeometricVertex(new long[] {values.length}, p);

        double calculatedP = myVertex.logPmf(values);
        double expectedP = 0.0;

        for (int value : values) {
            expectedP += getExpectedPmf(p, value);
        }

        assertEquals(expectedP, calculatedP, 1e-6);
    }

    private double getExpectedPmf(double p, int n) {
        return Math.log(Math.pow(1 - p, n - 1) * p);
    }

    @Test
    public void samplesMatchPMF() {
        double p = 0.2;

        GeometricVertex myVertex = new GeometricVertex(new long[] {1, 100000}, p);
        IntegerTensor samples = myVertex.sample();

        for (int i = 1; i < 100; i++) {
            checkCountMatchesPMF(myVertex, samples, i);
        }
    }

    private void checkCountMatchesPMF(GeometricVertex vertex, IntegerTensor samples, int n) {
        double expectedProportion = Math.exp(vertex.logPmf(n));
        double actualProportion = samples.asFlatList().stream()
            .filter(x -> x == n)
            .count() / (double)samples.getLength();

        assertEquals(expectedProportion, actualProportion, 5e-3);
    }
}

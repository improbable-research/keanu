package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.intgr.IntegerOverflowException;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BinomialVertexTest {

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double p = 0.25;
        int n = 5;

        BinomialVertex testPoissonVertex = new BinomialVertex(new int[]{1, N}, p, n);
        IntegerTensor samples = testPoissonVertex.sample();

        double mean = samples.toDouble().average();
        double std = samples.toDouble().standardDeviation();

        double epsilon = 0.1;
        assertEquals(n * p, mean, epsilon);
        assertEquals(n * p * (1 - p), std, epsilon);
    }

    @Test
    public void logPmfIsCorrectForKnownScalarValues() {

        double p = 0.25;
        int n = 5;

        BinomialVertex testPoissonVertex = new BinomialVertex(p, n);
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        for (int i = 0; i < n; i++) {
            double actual = testPoissonVertex.logPmf(i);
            double expected = distribution.logProbability(i);
            assertEquals(expected, actual, 1e-3);
        }
    }

    @Test
    public void logPmfIsCorrectForKnownVectorValues() {
        double p = 0.25;
        int n = 5;

        BinomialVertex testPoissonVertex = new BinomialVertex(new int[]{1, 2}, p, n);
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        double actual = testPoissonVertex.logPmf(new int[]{2, 3});
        double expected = distribution.logProbability(2) + distribution.logProbability(3);
        assertEquals(expected, actual, 1e-3);
    }

    @Test(expected = IntegerOverflowException.class)
    public void throwsExceptionOnOverflowWithVector() {
        BinomialVertex testPoissonVertex = new BinomialVertex(new int[]{1, 2}, 0.25, 15);
        testPoissonVertex.logPmf(14);
    }

    @Test(expected = IntegerOverflowException.class)
    public void throwsExceptionOnOverflowWithScalar() {
        BinomialVertex testPoissonVertex = new BinomialVertex(0.25, 15);
        testPoissonVertex.logPmf(14);
    }
}

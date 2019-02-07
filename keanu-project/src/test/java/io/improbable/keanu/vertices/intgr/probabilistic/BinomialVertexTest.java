package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BinomialVertexTest {

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double p = 0.25;
        int n = 5;

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{1, N}, p, n);
        IntegerTensor samples = testBinomialVertex.sample();

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

        BinomialVertex testBinomialVertex = new BinomialVertex(p, n);
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        for (int i = 0; i < n; i++) {
            double actual = testBinomialVertex.logPmf(i);
            double expected = distribution.logProbability(i);
            assertEquals(expected, actual, 1e-6);
        }
    }

    @Test
    public void logProbGraphIsCorrectForKnownScalarValues() {

        DoubleVertex p = ConstantVertex.of(0.25);
        IntegerVertex n = ConstantVertex.of(5);
        BinomialVertex binomialVertex = new BinomialVertex(p, n);
        LogProbGraph logProbGraph = binomialVertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, p, p.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, n, n.getValue());

        BinomialDistribution distribution = new BinomialDistribution(5, 0.25);

        for (int i = 0; i < 5; i++) {
            LogProbGraphValueFeeder.feedValueAndCascade(logProbGraph, binomialVertex, IntegerTensor.scalar(i));
            double expectedDensity = distribution.logProbability(i);
            LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
        }
    }

    @Test
    public void logPmfIsCorrectForKnownVectorValues() {
        double p = 0.25;
        int n = 100;
        int k1 = 20;
        int k2 = 80;

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{1, 2}, p, n);
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        double actual = testBinomialVertex.logPmf(new int[]{k1, k2});
        double expected = distribution.logProbability(k1) + distribution.logProbability(k2);
        assertEquals(expected, actual, 1e-6);
    }

    @Test
    public void logProbGraphIsCorrectForKnownVectorValues() {
        DoubleVertex p = ConstantVertex.of(0.25);
        IntegerVertex n = ConstantVertex.of(100);
        int k1 = 20;
        int k2 = 80;

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{1, 2}, p, n);
        LogProbGraph logProbGraph = testBinomialVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, p, p.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, n, n.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, testBinomialVertex, IntegerTensor.create(k1, k2));

        BinomialDistribution distribution = new BinomialDistribution(100, 0.25);
        double expectedDensity = distribution.logProbability(k1) + distribution.logProbability(k2);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }
}

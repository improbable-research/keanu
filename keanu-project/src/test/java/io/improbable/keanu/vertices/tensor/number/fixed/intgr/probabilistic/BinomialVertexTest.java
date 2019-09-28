package io.improbable.keanu.vertices.tensor.number.fixed.intgr.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.VertexVariationalMAP;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class BinomialVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double p = 0.25;
        int n = 5;

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{1, N}, p, n);
        IntegerTensor samples = testBinomialVertex.sample();

        double mean = samples.toDouble().mean().scalar();
        double std = samples.toDouble().standardDeviation().scalar();

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

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{2}, p, n);
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

        BinomialVertex testBinomialVertex = new BinomialVertex(new long[]{2}, p, n);
        LogProbGraph logProbGraph = testBinomialVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, p, p.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, n, n.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, testBinomialVertex, IntegerTensor.create(k1, k2));

        BinomialDistribution distribution = new BinomialDistribution(100, 0.25);
        double expectedDensity = distribution.logProbability(k1) + distribution.logProbability(k2);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueP = 0.7;

        List<DoubleVertex> p = new ArrayList<>();
        p.add(ConstantVertex.of(trueP));

        List<DoubleVertex> latents = new ArrayList<>();
        UniformVertex latentP = new UniformVertex(0.01, 10.0);
        latentP.setAndCascade(DoubleTensor.scalar(0.2));
        latents.add(latentP);

        int numSamples = 500;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new BinomialVertex(new long[]{numSamples}, hyperParams.get(0), ConstantVertex.of(2)),
            p,
            latents,
            1e-3,
            random
        );
    }

    @Test
    public void inferBatchHyperParamsFromSamples() {

        DoubleTensor trueP = DoubleTensor.create(0.7, 0.35);

        List<DoubleVertex> p = new ArrayList<>();
        p.add(ConstantVertex.of(trueP));

        List<DoubleVertex> latents = new ArrayList<>();
        UniformVertex latentP = new UniformVertex(0.01, 10.0);
        latentP.setAndCascade(DoubleTensor.create(0.2, 0.8));
        latents.add(latentP);

        int numSamples = 500;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new BinomialVertex(new long[]{numSamples, 2, 2}, hyperParams.get(0), ConstantVertex.of(IntegerTensor.create(0, 1, 2, 3).reshape(2, 2))),
            p,
            latents,
            random
        );
    }
}

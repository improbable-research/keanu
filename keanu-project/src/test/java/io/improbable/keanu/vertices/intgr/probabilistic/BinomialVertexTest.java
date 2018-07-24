package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.junit.Assert.assertEquals;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.BuilderParameterException;
import io.improbable.keanu.vertices.MissingParameterException;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class BinomialVertexTest {

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double p = 0.25;
        int n = 5;

        BinomialVertex testPoissonVertex = new DistributionVertexBuilder()
            .shaped(1, N)
            .withInput(ParameterName.P, p)
            .withInput(ParameterName.N, n)
            .binomial();
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

        BinomialVertex testPoissonVertex = new DistributionVertexBuilder()
            .withInput(ParameterName.P, p)
            .withInput(ParameterName.N, n)
            .binomial();
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        for (int i = 0; i < n; i++) {
            double actual = testPoissonVertex.logProb(IntegerTensor.scalar(i));
            double expected = distribution.logProbability(i);
            assertEquals(expected, actual, 1e-3);
        }
    }

    @Test
    public void logPmfIsCorrectForKnownVectorValues() {
        double p = 0.25;
        int n = 50;
        int k1 = 20;
        int k2 = 30;

        BinomialVertex testPoissonVertex = new DistributionVertexBuilder()
            .shaped(1, 2)
            .withInput(ParameterName.P, p)
            .withInput(ParameterName.N, n)
            .binomial();
        BinomialDistribution distribution = new BinomialDistribution(n, p);

        double actual = testPoissonVertex.logProb(IntegerTensor.create(new int[]{k1, k2}));
        double expected = distribution.logProbability(k1) + distribution.logProbability(k2);
        assertEquals(expected, actual, 1e-3);
    }

    @Test(expected = BuilderParameterException.class)
    public void itThrowsIfTheInputDimensionsDontMatch() {
        new DistributionVertexBuilder()
            .withInput(ParameterName.P, new ConstantDoubleVertex(new double[] {1.,2.,3.}))
            .withInput(ParameterName.N, new ConstantIntegerVertex(new int[] {1,2}))
            .binomial();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouHaventSetParameterN() {
        new DistributionVertexBuilder()
            .shaped(1,2,3)
            .withInput(ParameterName.P, new ConstantDoubleVertex(new double[] {1.,2.,3.}))
            .binomial();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouHaventSetParameterP() {
        new DistributionVertexBuilder()
            .shaped(1,2,3)
            .withInput(ParameterName.N, new ConstantIntegerVertex(new int[] {1,2}))
            .binomial();
    }
}

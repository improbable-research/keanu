package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MCMCTestDistributions {

    public static BayesNet createSimpleGaussian(double mu, double sigma, Random random) {
        GaussianVertex A = new GaussianVertex(mu, sigma);
        A.setAndCascade(mu + 0.5 * sigma);
        BayesNet bayesNet = new BayesNet(A.getConnectedGraph());
        return bayesNet;
    }

    public static void samplesMatchSimpleGaussian(double mu, double sigma, List<Double> samples) {

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        assertEquals(mu, stats.getMean(), 0.05);
        assertEquals(sigma, stats.getStandardDeviation(), 0.1);
    }

    public static BayesNet createSumOfGaussianDistribution(double mu, double sigma, double observedSum, Random random) {

        DoubleVertex A = new GaussianVertex(mu, sigma);
        DoubleVertex B = new GaussianVertex(mu, sigma);

        DoubleVertex C = new GaussianVertex(A.plus(B), 1.0);
        C.observe(observedSum);

        A.setAndCascade(mu);
        B.setAndCascade(mu);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, C));
        return bayesNet;
    }

    public static void samplesMatchesSumOfGaussians(double expected, List<Double> sampleA, List<Double> samplesB) {

        OptionalDouble averagePosteriorA = sampleA.stream()
                .mapToDouble(sample -> sample)
                .average();

        OptionalDouble averagePosteriorB = samplesB.stream()
                .mapToDouble(sample -> sample)
                .average();

        assertEquals(expected, averagePosteriorA.getAsDouble() + averagePosteriorB.getAsDouble(), 0.1);
    }

    public static BayesNet create2DDonutDistribution(Random random) {
        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleVertex B = new GaussianVertex(0, 1);

        DoubleVertex D = new GaussianVertex((A.multiply(A)).plus(B.multiply(B)), 0.03);
        D.observe(0.5);

        A.setAndCascade(Math.sqrt(0.5));
        B.setAndCascade(0.0);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, D));
        return bayesNet;
    }

    public static void samplesMatch2DDonut(List<Double> samplesA, List<Double> samplesB) {

        boolean topOfDonut, rightOfDonut, bottomOfDonut, leftOfDonut, middleOfDonut;
        topOfDonut = rightOfDonut = bottomOfDonut = leftOfDonut = middleOfDonut = false;

        for (int i = 0; i < samplesA.size(); i++) {
            double sampleFromA = samplesA.get(i);
            double sampleFromB = samplesB.get(i);

            if (sampleFromA > -0.2 && sampleFromA < 0.2 && sampleFromB > 0.6 && sampleFromB < 0.8) {
                topOfDonut = true;
            } else if (sampleFromA > 0.6 && sampleFromA < 0.8 && sampleFromB > -0.2 && sampleFromB < 0.2) {
                rightOfDonut = true;
            } else if (sampleFromA > -0.2 && sampleFromA < 0.2 && sampleFromB > -0.8 && sampleFromB < -0.6) {
                bottomOfDonut = true;
            } else if (sampleFromA > -0.8 && sampleFromA < -0.6 && sampleFromB > -0.2 && sampleFromB < 0.2) {
                leftOfDonut = true;
            } else if (sampleFromA > 0.35 && sampleFromA < 0.45 && sampleFromB > 0.35 && sampleFromB < 0.45) {
                middleOfDonut = true;
            }
        }

        assertTrue(topOfDonut && rightOfDonut && bottomOfDonut && leftOfDonut && !middleOfDonut);
    }

}

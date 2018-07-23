package io.improbable.keanu.algorithms.mcmc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class MCMCTestDistributions {

    public static BayesianNetwork createSimpleGaussian(double mu, double sigma, KeanuRandom random) {
        GaussianVertex A = new DistributionVertexBuilder()
            .shaped(new int[]{2, 1})
            .withInput(ParameterName.MU, mu)
            .withInput(ParameterName.SIGMA, sigma)
            .gaussian();
        A.setAndCascade(A.sample(random));
        return new BayesianNetwork(A.getConnectedGraph());
    }

    public static void samplesMatchSimpleGaussian(double mu, double sigma, List<DoubleTensor> samples) {

        int[] shape = samples.get(0).getShape();

        DoubleTensor summed = samples.stream()
            .reduce(DoubleTensor.zeros(shape), DoubleTensor::plusInPlace);

        DoubleTensor averages = summed.divInPlace(samples.size());

        DoubleTensor sumDiffSquared = samples.stream()
            .reduce(
                DoubleTensor.zeros(shape),
                (acc, tensor) -> acc.plusInPlace(tensor.minus(averages).powInPlace(2))
            );

        double[] standardDeviations = sumDiffSquared.div(samples.size() - 1).pow(0.5).asFlatDoubleArray();
        double[] means = averages.asFlatDoubleArray();

        for (int i = 0; i < means.length; i++) {
            assertEquals(mu, means[i], 0.05);
            assertEquals(sigma, standardDeviations[i], 0.1);
        }
    }

    public static BayesianNetwork createSumOfGaussianDistribution(double mu, double sigma, double observedSum) {

        GaussianVertex A = VertexOfType.gaussian(mu, sigma);
        GaussianVertex B = VertexOfType.gaussian(mu, sigma);

        GaussianVertex C = VertexOfType.gaussian(A.plus(B), ConstantVertex.of(1.0));
        C.observe(DoubleTensor.scalar(observedSum));

        A.setValue(mu);
        B.setAndCascade(mu);

        return new BayesianNetwork(Arrays.asList(A, B, C));
    }

    public static void samplesMatchesSumOfGaussians(double expected, List<DoubleTensor> sampleA, List<DoubleTensor> samplesB) {

        OptionalDouble averagePosteriorA = sampleA.stream()
            .flatMapToDouble(tensor -> Arrays.stream(tensor.asFlatDoubleArray()))
            .average();

        OptionalDouble averagePosteriorB = samplesB.stream()
            .flatMapToDouble(tensor -> Arrays.stream(tensor.asFlatDoubleArray()))
            .average();

        assertEquals(expected, averagePosteriorA.getAsDouble() + averagePosteriorB.getAsDouble(), 0.1);
    }

    public static BayesianNetwork create2DDonutDistribution() {
        GaussianVertex A = VertexOfType.gaussian(0., 1.);
        GaussianVertex B = VertexOfType.gaussian(0., 1.);

        GaussianVertex D = VertexOfType.gaussian((A.multiply(A)).plus(B.multiply(B)), ConstantVertex.of(0.03));
        D.observe(DoubleTensor.scalar(0.5));

        A.setValue(Math.sqrt(0.5));
        B.setAndCascade(0.0);

        return new BayesianNetwork(Arrays.asList(A, B, D));
    }

    public static void samplesMatch2DDonut(List<DoubleTensor> samplesA, List<DoubleTensor> samplesB) {

        boolean topOfDonut, rightOfDonut, bottomOfDonut, leftOfDonut, middleOfDonut;
        topOfDonut = rightOfDonut = bottomOfDonut = leftOfDonut = middleOfDonut = false;

        for (int i = 0; i < samplesA.size(); i++) {
            double sampleFromA = samplesA.get(i).scalar();
            double sampleFromB = samplesB.get(i).scalar();

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

package io.improbable.keanu.algorithms.variational;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract;

public class KDEApproximationTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private static final double DELTA = 0.1;

    public DoubleVertexSamples generateGaussianSamples(double mu, double sigma, int nSamples) {
        DoubleVertex gaussian = new GaussianVertex(mu, sigma);
        BayesianNetwork network = new BayesianNetwork(gaussian.getConnectedGraph());
        return MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(network, Collections.singletonList(gaussian), nSamples)
            .getDoubleTensorSamples(gaussian);
    }

    public static void isCloseMostOfTheTime(DoubleTensor expected, DoubleTensor approximated, double correctPercentage, double delta) {
        double nCorrect = 0.;
        for (int i = 0; i < approximated.getLength(); i++) {
            Double approximateValue = approximated.asFlatList().get(i);
            Double expectedValue = expected.asFlatList().get(i);
            if (Math.abs(approximateValue - expectedValue) < delta) {
                nCorrect++;
            }
        }
        assertTrue(String.format("Only %f out of %d correct!", nCorrect, expected.asFlatList().size()), nCorrect / expected.asFlatList().size() > correctPercentage);
    }

    @Test
    public void matchesKnownLogDensityOfScalar() {
        double mu = 1.;
        double sigma = 1.;
        double correctPercentage = 0.8;
        double delta = 0.1;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 100000);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        DoubleTensor x = DoubleTensor.linspace(-3., 3., 100);
        DoubleTensor gaussianLogPdf = Gaussian.withParameters(
            DoubleTensor.scalar(mu),
            DoubleTensor.scalar(sigma)
        ).logProb(x);

        DoubleTensor expectedPdf = DoubleTensor.create(Math.E, x.getShape()).pow(gaussianLogPdf);
        DoubleTensor approximatedPdf = KDE.pdf(x);

        isCloseMostOfTheTime(expectedPdf, approximatedPdf, correctPercentage, delta);
    }

    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        double mu = 1.;
        double sigma = 1.;
        double correctPercentage = 0.9;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 1000000);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        DoubleTensor xTensor = DoubleTensor.linspace(-1. + mu, 1. + mu, 10);
        Diffs diffLog = Gaussian.withParameters(
            DoubleTensor.scalar(mu),
            DoubleTensor.scalar(sigma)
        ).dLogProb(xTensor);

        DoubleTensor approximateDerivative = KDE.dLogPdf(xTensor, KDE).get(KDE.getId());
        DoubleTensor expectedDerivative = diffLog.get(Diffs.X).getValue();
        isCloseMostOfTheTime(expectedDerivative, approximateDerivative, correctPercentage, DELTA);
    }

    @Test
    public void dLogPdfForMultipleInputsTest() {
        double mu = 0.;
        double sigma = 1.;
        double DELTA = 0.1;
        double correctPercentage = 0.8;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 100000);
        GaussianVertex gaussian = new GaussianVertex(mu, sigma);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        DoubleTensor x = DoubleTensor.linspace(-1., 1., 100);
        DoubleTensor approximateDerivative = KDE.dLogPdf(x, KDE).get(KDE.getId());
        DoubleTensor expectedDerivative = gaussian.dLogProb(x).get(gaussian.getId());

        isCloseMostOfTheTime(expectedDerivative, approximateDerivative, correctPercentage, DELTA);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {
        double mu = 10.;
        double sigma = .5;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 10000);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        double from = -3;
        double to = 3;
        double bucketSize = 0.1;

        ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod(
            KDE, from, to, bucketSize, 1e-2, KeanuRandom.getDefaultRandom(), 1000
        );
    }

    @Test
    public void resamplingTest() {
        double mu = 10.;
        double sigma = .5;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 10000);

        KDEVertex KDE = GaussianKDE.approximate(samples);
        KDEVertex resampledKDE = GaussianKDE.approximate(samples);

        int nSamples = 1000;
        resampledKDE.resample(nSamples, KeanuRandom.getDefaultRandom());
        assertEquals(1, resampledKDE.getSampleShape()[0]);
        assertEquals(nSamples, resampledKDE.getSampleShape()[1]);
    }

    @Test(expected = IllegalArgumentException.class)
    public void handlingNonScalarSamplesTest() {
        List<DoubleTensor> badSamplesList = Collections.singletonList(DoubleTensor.create(new double[]{1, 2, 3}));

        DoubleVertexSamples badSamples = new DoubleVertexSamples(badSamplesList);
        KDEVertex KDE = GaussianKDE.approximate(badSamples);
        throw new AssertionError("approximate did not throw a IllegalArgumentException!");
    }
}

package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.KDEVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static io.improbable.keanu.tensor.TensorMatchers.hasShape;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class KDEApproximationTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private static final double DELTA = 0.1;

    public DoubleVertexSamples generateGaussianSamples(double mu, double sigma, int nSamples) {
        GaussianVertex gaussian = new GaussianVertex(mu, sigma);
        List<DoubleTensor> samples = new ArrayList<>(nSamples);

        for (int i = 0; i < nSamples; i++) {
            samples.add(gaussian.sample());
        }

        return new DoubleVertexSamples(samples);
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
        assertTrue(String.format("Only %f out of %d correct!", nCorrect, expected.asFlatList().size()), nCorrect / expected.asFlatList().size() >= correctPercentage);
    }

    @Category(Slow.class)
    @Test
    public void matchesKnownLogDensityOfScalar() {
        double mu = 1.;
        double sigma = 1.;
        double correctPercentage = 0.8;
        double delta = 0.1;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 50000);

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

    @Category(Slow.class)
    @Test
    public void matchesKnownDerivativeLogDensityOfScalar() {
        double mu = 1.;
        double sigma = 1.;
        double correctPercentage = 0.9;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 10000);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        DoubleTensor xTensor = DoubleTensor.linspace(-1. + mu, 1. + mu, 10);
        Diffs diffLog = Gaussian.withParameters(
            DoubleTensor.scalar(mu),
            DoubleTensor.scalar(sigma)
        ).dLogProb(xTensor);

        DoubleTensor approximateDerivative = KDE.dLogPdf(xTensor, KDE).get(KDE);
        DoubleTensor expectedDerivative = diffLog.get(Diffs.X).getValue();
        isCloseMostOfTheTime(expectedDerivative, approximateDerivative, correctPercentage, DELTA);
    }

    @Category(Slow.class)
    @Test
    public void dLogPdfForMultipleInputsTest() {
        double mu = 0.;
        double sigma = 1.;
        double DELTA = 0.1;
        double correctPercentage = 0.9;

        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 10000);
        GaussianVertex gaussian = new GaussianVertex(mu, sigma);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        DoubleTensor x = DoubleTensor.linspace(-1., 1., 100);
        DoubleTensor approximateDerivative = KDE.dLogPdf(x, KDE).get(KDE);
        DoubleTensor expectedDerivative = gaussian.dLogProb(x, gaussian).get(gaussian);

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
    public void youCanSampleAScalarMultipleTimes() {
        double mu = 10.;
        double sigma = .5;
        DoubleVertexSamples samples = generateGaussianSamples(mu, sigma, 2);

        KDEVertex KDE = GaussianKDE.approximate(samples);

        int numSamples = 100;
        DoubleTensor sample = KDE.sample(numSamples, KeanuRandom.getDefaultRandom());
        assertThat(sample, hasShape(100));
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
        assertEquals(nSamples, resampledKDE.getSampleShape()[0]);
    }

    @Test(expected = IllegalArgumentException.class)
    public void handlingNonScalarSamplesTest() {
        List<DoubleTensor> badSamplesList = Collections.singletonList(DoubleTensor.create(new double[]{1, 2, 3}));

        DoubleVertexSamples badSamples = new DoubleVertexSamples(badSamplesList);
        KDEVertex KDE = GaussianKDE.approximate(badSamples);
        throw new AssertionError("approximate did not throw a IllegalArgumentException!");
    }
}

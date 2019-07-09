package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.mcmc.testcases.MCMCTestDistributions;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.everyItem;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

public class NUTSTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Category(Slow.class)
    @Test
    public void canRecordStatisticsFromSamples() {
        double mu = 0.0;
        double sigma = 1.0;
        double initStepSize = 1;
        int maxTreeHeight = 4;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptStepSizeEnabled(false)
            .initialStepSize(initStepSize)
            .maxTreeHeight(maxTreeHeight)
            .saveStatistics(true)
            .build();

        nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            2
        );

        Statistics statistics = nuts.getStatistics();

        List<Double> stepSize = statistics.get(NUTS.Metrics.STEPSIZE);
        List<Double> logProb = statistics.get(NUTS.Metrics.LOG_PROB);
        List<Double> meanTreeAccept = statistics.get(NUTS.Metrics.MEAN_TREE_ACCEPT);
        List<Double> treeSize = statistics.get(NUTS.Metrics.TREE_SIZE);

        assertThat(stepSize, contains(initStepSize, initStepSize));
        assertThat(logProb, everyItem(lessThan(0.)));
        assertThat(meanTreeAccept, everyItem(both(greaterThanOrEqualTo(0.)).and(lessThanOrEqualTo(1.))));
        assertThat(treeSize, everyItem(lessThan(Math.pow(2, maxTreeHeight))));
    }

    @Category(Slow.class)
    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        int sampleCount = 300;
        NUTS nuts = NUTS.builder()
            .adaptCount(sampleCount / 4)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            sampleCount
        );

        IVertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);
        List<DoubleTensor> nutsSamples = posteriorSamples.get(vertex).asList();

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, nutsSamples);
    }

    @Test
    public void samplesContinuousPrior() {

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46., 15.0);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(bayesNet);

        int sampleCount = 500;
        NUTS nuts = NUTS.builder()
            .adaptCount(sampleCount)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            sampleCount
        ).drop(sampleCount / 4);

        IVertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        IVertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Category(Slow.class)
    @Test
    public void samplesFromDonut() {
        BayesianNetwork donutBayesNet = MCMCTestDistributions.create2DDonutDistribution();
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(donutBayesNet);

        int sampleCount = 1000;
        NUTS nuts = NUTS.builder()
            .adaptCount(sampleCount)
            .build();

        NetworkSamples samples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            sampleCount
        );

        IVertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        IVertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }

    @Test
    public void canDefaultToSettingsInBuilder() {

        GaussianVertex A = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(net);

        NUTS nuts = NUTS.builder()
            .build();

        assertTrue(nuts.getAdaptCount() > 0);
        assertTrue(nuts.getTargetAcceptanceProb() > 0);
        assertNotNull(nuts.getRandom());

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            2
        );

        assertFalse(posteriorSamples.get(A).asList().isEmpty());
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsIfStartingPositionIsZeroProbability() {

        GaussianVertex A = new HalfGaussianVertex(1.0);
        A.setValue(-1.0);

        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());

        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(net);

        NUTS nuts = NUTS.builder()
            .build();

        nuts.getPosteriorSamples(model, 2);
    }

    @Category(Slow.class)
    @Test
    /*
     * This test assumes the functional logic of NUTS has not been changed.
     * It simply checks that the samples produced are identical to a previous build.
     * If this test fails, it may still be functionally correct. Please make sure
     * you understand why the samples have changed before accepting them by updating these
     * magic numbers.
     */
    public void checksSamplesAgainstMagicNumbers() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3);
        KeanuProbabilisticModel model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptCount(5)
            .targetAcceptanceProb(0.65)
            .maxTreeHeight(10)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            20
        );

        IVertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        List<DoubleTensor> samples = posteriorSamples.get(vertex).asList();

        double epsilon = 1e-9;

        //Use this if you're sure you want to accept a change in the nuts walk given the above setup
        //for (int i = 0; i < samples.size(); i++) {
        //    System.out.println("Assert.assertEquals(" + samples.get(i).scalar() + ", samples.get(" + i + ").scalar(), epsilon);");
        //}

        Assert.assertEquals(2.655769904078054, samples.get(0).scalar(), epsilon);
        Assert.assertEquals(2.655769904078054, samples.get(1).scalar(), epsilon);
        Assert.assertEquals(-1.0165440910526553, samples.get(2).scalar(), epsilon);
        Assert.assertEquals(-1.0165440910526553, samples.get(3).scalar(), epsilon);
        Assert.assertEquals(-1.1033732018241589, samples.get(4).scalar(), epsilon);
        Assert.assertEquals(-1.5926775601489656, samples.get(5).scalar(), epsilon);
        Assert.assertEquals(-1.5926775601489656, samples.get(6).scalar(), epsilon);
        Assert.assertEquals(0.06269495259568081, samples.get(7).scalar(), epsilon);
        Assert.assertEquals(0.06269495259568081, samples.get(8).scalar(), epsilon);
        Assert.assertEquals(0.06269495259568081, samples.get(9).scalar(), epsilon);
        Assert.assertEquals(-1.9067350000491319, samples.get(10).scalar(), epsilon);
        Assert.assertEquals(-0.16217717982790925, samples.get(11).scalar(), epsilon);
        Assert.assertEquals(-0.16217717982790925, samples.get(12).scalar(), epsilon);
        Assert.assertEquals(-1.7749397804101408, samples.get(13).scalar(), epsilon);
        Assert.assertEquals(-1.95774120776884, samples.get(14).scalar(), epsilon);
        Assert.assertEquals(1.1167543688694772, samples.get(15).scalar(), epsilon);
        Assert.assertEquals(-2.3326358434845753, samples.get(16).scalar(), epsilon);
        Assert.assertEquals(-2.425061791760318, samples.get(17).scalar(), epsilon);
        Assert.assertEquals(1.3182493371930095, samples.get(18).scalar(), epsilon);
        Assert.assertEquals(-1.3168733581639154, samples.get(19).scalar(), epsilon);

    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateAdapStepSizeCount() {
        NUTS.builder().adaptCount(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateInitialStepSize() {
        NUTS.builder().initialStepSize(-0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateTargetAcceptanceUpper() {
        NUTS.builder().targetAcceptanceProb(1.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateTargetAcceptanceLower() {
        NUTS.builder().targetAcceptanceProb(-0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateMaxEnergyChange() {
        NUTS.builder().maxEnergyChange(-0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void doesValidateMaxTreeHeight() {
        NUTS.builder().maxTreeHeight(0);
    }

}

package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.mcmc.MCMCTestDistributions;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

public class NUTSTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Category(Slow.class)
    @Test
    public void canRecordStatisticsFromSamples() {
        double mu = 0.0;
        double sigma = 1.0;
        double initStepSize = 1;
        int maxTreeHeight = 4;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3, random);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptEnabled(false)
            .initialStepSize(initStepSize)
            .random(random)
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
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3, random);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptCount(2000)
            .random(random)
            .targetAcceptanceProb(0.65)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            2000
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList(), 0.1);
    }

    @Test
    public void samplesContinuousPrior() {

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46., 15.0);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(bayesNet);

        int sampleCount = 6000;
        NUTS nuts = NUTS.builder()
            .adaptCount(sampleCount)
            .maxTreeHeight(4)
            .targetAcceptanceProb(0.6)
            .random(random)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            sampleCount
        ).drop(sampleCount / 4);

        Vertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Category(Slow.class)
    @Test
    public void samplesFromDonut() {
        BayesianNetwork donutBayesNet = MCMCTestDistributions.create2DDonutDistribution();
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(donutBayesNet);

        NUTS nuts = NUTS.builder()
            .adaptCount(1000)
            .random(random)
            .build();

        NetworkSamples samples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            1000
        );

        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }

    @Test
    public void canDefaultToSettingsInBuilder() {

        GaussianVertex A = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100, random);
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

    @Category(Slow.class)
    @Test
    /**
     * This test assumes the functional logic of NUTS has not been changed.
     * It simply checks that the samples produced are identical to a previous build.
     */
    public void checksSamplesAgainstMagicNumbers() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3, random);
        KeanuProbabilisticModel model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptCount(5)
            .random(random)
            .targetAcceptanceProb(0.65)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            20
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        List<DoubleTensor> samples = posteriorSamples.get(vertex).asList();

        Assert.assertEquals(3.0, samples.get(0).scalar(), 1e-9);
        Assert.assertEquals(3.0, samples.get(1).scalar(), 1e-9);
        Assert.assertEquals(0.9374092571432446, samples.get(2).scalar(), 1e-9);
        Assert.assertEquals(0.05720950629236243, samples.get(3).scalar(), 1e-9);
        Assert.assertEquals(0.33119352888492626, samples.get(4).scalar(), 1e-9);
        Assert.assertEquals(0.9124861769925321, samples.get(19).scalar(), 1e-9);

    }
}

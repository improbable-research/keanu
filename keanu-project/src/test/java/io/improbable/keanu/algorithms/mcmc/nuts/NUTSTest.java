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
import io.improbable.keanu.vertices.Vertex;
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
            .adaptEnabled(false)
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

        int sampleCount = 1000;
        NUTS nuts = NUTS.builder()
            .adaptCount(sampleCount)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            sampleCount
        ).drop(sampleCount / 4);

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList(), 0.1);
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

        Vertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

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

        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

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
    /**
     * This test assumes the functional logic of NUTS has not been changed.
     * It simply checks that the samples produced are identical to a previous build.
     */
    public void checksSamplesAgainstMagicNumbers() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, 3);
        KeanuProbabilisticModel model = new KeanuProbabilisticModelWithGradient(simpleGaussian);

        NUTS nuts = NUTS.builder()
            .adaptCount(5)
            .targetAcceptanceProb(0.65)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            model,
            model.getLatentVariables(),
            20
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        List<DoubleTensor> samples = posteriorSamples.get(vertex).asList();

        double epsilon = 1e-9;

        Assert.assertEquals(3.0, samples.get(0).scalar(), epsilon);
        Assert.assertEquals(3.0, samples.get(1).scalar(), epsilon);
        Assert.assertEquals(-3.340614624594811, samples.get(2).scalar(), epsilon);
        Assert.assertEquals(-1.6600011056731434, samples.get(3).scalar(), epsilon);
        Assert.assertEquals(-0.25252167633695866, samples.get(4).scalar(), epsilon);
        Assert.assertEquals(1.3636808333859607, samples.get(5).scalar(), epsilon);
        Assert.assertEquals(0.7124949458410073, samples.get(6).scalar(), epsilon);
        Assert.assertEquals(1.5925606313564984, samples.get(7).scalar(), epsilon);
        Assert.assertEquals(1.5925606313564984, samples.get(8).scalar(), epsilon);
        Assert.assertEquals(1.200842781172803, samples.get(9).scalar(), epsilon);
        Assert.assertEquals(1.200842781172803, samples.get(10).scalar(), epsilon);
        Assert.assertEquals(0.427385289391189, samples.get(11).scalar(), epsilon);
        Assert.assertEquals(-1.8112210534316109, samples.get(12).scalar(), epsilon);
        Assert.assertEquals(0.40880399903885367, samples.get(13).scalar(), epsilon);
        Assert.assertEquals(0.736114340288887, samples.get(14).scalar(), epsilon);
        Assert.assertEquals(0.3642122014037824, samples.get(15).scalar(), epsilon);
        Assert.assertEquals(-0.9919123539994699, samples.get(16).scalar(), epsilon);
        Assert.assertEquals(-0.4056102006678568, samples.get(17).scalar(), epsilon);
        Assert.assertEquals(-0.27304744507279877, samples.get(18).scalar(), epsilon);
        Assert.assertEquals(0.5661014922937297, samples.get(19).scalar(), epsilon);

    }
}

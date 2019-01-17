package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.mcmc.MCMCTestDistributions;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModelWithGradient;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
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

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46., 18.0);
        ProbabilisticModelWithGradient model = new KeanuProbabilisticModelWithGradient(bayesNet);

        int sampleCount = 2000;
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
}

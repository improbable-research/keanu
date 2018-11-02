package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class HamiltonianTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, KeanuRandom.getDefaultRandom());

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(10)
            .stepSize(0.4)
            .build();

        NetworkSamples posteriorSamples = hmc.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            1000
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList());
    }

    @Test
    public void samplesContinuousPrior() {

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.0);

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(20)
            .stepSize(0.1)
            .build();

        NetworkSamples posteriorSamples = hmc.getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            2000
        );

        Vertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Test
    public void samplesFromDonut() {

        BayesianNetwork donutBayesNet = MCMCTestDistributions.create2DDonutDistribution();

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(15)
            .stepSize(0.02)
            .build();

        NetworkSamples samples = hmc.getPosteriorSamples(
            donutBayesNet,
            donutBayesNet.getLatentVertices(),
            2500
        );

        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }


    @Test
    public void canDefaultToSettingsInBuilderAndIsConfigurableAfterBuilding() {

        GaussianVertex A = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100, KeanuRandom.getDefaultRandom());

        Hamiltonian hmc = Hamiltonian.builder()
            .build();

        assertTrue(hmc.getLeapFrogCount() > 0);
        assertTrue(hmc.getStepSize() > 0);
        assertNotNull(hmc.getRandom());

        NetworkSamples posteriorSamples = hmc.getPosteriorSamples(
            net,
            net.getLatentVertices(),
            2
        );

        hmc.setRandom(null);
        assertNull(hmc.getRandom());

        assertFalse(posteriorSamples.get(A).asList().isEmpty());
    }

    @Test
    public void canStreamSamples() {

        Hamiltonian algo = Hamiltonian.withDefaultConfig();

        int sampleCount = 1000;
        int dropCount = 100;
        int downSampleInterval = 1;
        GaussianVertex A = new GaussianVertex(0, 1);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        double averageA = algo.generatePosteriorSamples(network, network.getLatentVertices())
            .dropCount(dropCount)
            .downSampleInterval(downSampleInterval)
            .stream()
            .limit(sampleCount)
            .mapToDouble(networkState -> networkState.get(A).scalar())
            .average().getAsDouble();

        assertEquals(0.0, averageA, 0.1);
    }
}

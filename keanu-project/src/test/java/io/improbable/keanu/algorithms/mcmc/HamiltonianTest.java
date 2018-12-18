package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class HamiltonianTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Category(Slow.class)
    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        KeanuRandom random = KeanuRandom.getDefaultRandom();
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, random.nextGaussian(0, 1), random);

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(20)
            .stepSize(0.05)
            .build();

        NetworkSamples posteriorSamples = hmc.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            2000
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList(), 0.1);
    }

    @Test
    public void samplesContinuousPrior() {

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.0, 20.0);

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(20)
            .stepSize(0.1)
            .build();

        NetworkSamples posteriorSamples = hmc.getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            600
        );

        Vertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Category(Slow.class)
    @Test
    public void samplesFromDonut() {

        BayesianNetwork donutBayesNet = MCMCTestDistributions.create2DDonutDistribution();
        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(15)
            .stepSize(0.02)
            .build();

        Map<Vertex, List<DoubleTensor>> samples = new HashMap<>();

        hmc.generatePosteriorSamples(donutBayesNet, donutBayesNet.getLatentVertices())
            .stream()
            .limit(350)
            .forEach(x -> {
                samples.computeIfAbsent(A, k -> new ArrayList<>()).add(x.get(A));
                samples.computeIfAbsent(B, k -> new ArrayList<>()).add(x.get(B));
            });

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A), samples.get(B));
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

        Hamiltonian hmc = Hamiltonian.builder()
            .leapFrogCount(10)
            .stepSize(0.4)
            .build();

        double mu = 0;
        double sigma = 1;
        GaussianVertex A = new GaussianVertex(mu, sigma);
        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        List<DoubleTensor> samples = hmc.generatePosteriorSamples(network, network.getLatentVertices())
            .downSampleInterval(1)
            .stream()
            .limit(500)
            .map(x -> x.get(A))
            .collect(Collectors.toList());

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, samples, 0.3);
    }

}

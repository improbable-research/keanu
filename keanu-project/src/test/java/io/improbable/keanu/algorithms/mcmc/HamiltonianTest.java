package io.improbable.keanu.algorithms.mcmc;

import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
}

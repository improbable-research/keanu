package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;

public class HamiltonianTest {

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesNetDoubleAsContinuous simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, random);

        NetworkSamples posteriorSamples = Hamiltonian.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            1000,
            20,
            0.15,
            random
        );

        Vertex<Double> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList());
    }

    @Test
    public void samplesContinuousPrior() {

        BayesNetDoubleAsContinuous bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.0, random);

        NetworkSamples posteriorSamples = Hamiltonian.getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            2000,
            20,
            0.1,
            random
        );

        Vertex<Double> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<Double> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Test
    public void samplesFromDonut() {

        BayesNetDoubleAsContinuous donutBayesNet = MCMCTestDistributions.create2DDonutDistribution(random);

        NetworkSamples samples = Hamiltonian.getPosteriorSamples(
            donutBayesNet,
            donutBayesNet.getLatentVertices(),
            2500,
            10,
            0.05,
            random
        );

        Vertex<Double> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<Double> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }
}

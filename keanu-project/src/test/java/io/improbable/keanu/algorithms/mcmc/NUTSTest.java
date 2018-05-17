package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class NUTSTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        BayesNet simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, random);

        NetworkSamples posteriorSamples = NUTS.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            1000,
            0.3,
            random
        );

        Vertex<Double> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList());
    }

    @Test
    public void samplesContinuousPrior() {

        BayesNet bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.0, random);

        NetworkSamples posteriorSamples = NUTS.getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            2000,
            0.1,
            random
        );

        Vertex<Double> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<Double> B = bayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Test
    public void samplesFromDonut() {
        BayesNet donutBayesNet = MCMCTestDistributions.create2DDonutDistribution(random);

        NetworkSamples samples = NUTS.getPosteriorSamples(
            donutBayesNet,
            donutBayesNet.getLatentVertices(),
            1000,
            0.05,
            random
        );

        Vertex<Double> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<Double> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }
}


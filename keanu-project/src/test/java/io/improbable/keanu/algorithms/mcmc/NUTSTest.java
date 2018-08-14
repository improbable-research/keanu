package io.improbable.keanu.algorithms.mcmc;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
        BayesianNetwork simpleGaussian = MCMCTestDistributions.createSimpleGaussian(mu, sigma, random);

        NUTS nuts = NUTS.builder()
            .adaptCount(50)
            .random(random)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            1000
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        MCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList());
    }

    @Test
    public void samplesContinuousPrior() {

        BayesianNetwork bayesNet = MCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.);

        NUTS nuts = NUTS.builder()
            .adaptCount(100)
            .random(random)
            .build();

        NetworkSamples posteriorSamples = nuts.getPosteriorSamples(
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

        NUTS nuts = NUTS.builder()
            .adaptCount(100)
            .random(random)
            .build();

        NetworkSamples samples = nuts.getPosteriorSamples(
            donutBayesNet,
            donutBayesNet.getLatentVertices(),
            1000
        );

        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        MCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }
}


package io.improbable.keanu.algorithms.tensormcmc;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.TensorBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

public class TensorHamiltonianTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesGaussian() {
        double mu = 0.0;
        double sigma = 1.0;
        TensorBayesNet simpleGaussian = TensorMCMCTestDistributions.createSimpleGaussian(mu, sigma, random);

        NetworkSamples posteriorSamples = TensorHamiltonian.getPosteriorSamples(
            simpleGaussian,
            simpleGaussian.getLatentVertices(),
            1000,
            10,
            0.4,
            random
        );

        Vertex<DoubleTensor> vertex = simpleGaussian.getContinuousLatentVertices().get(0);

        TensorMCMCTestDistributions.samplesMatchSimpleGaussian(mu, sigma, posteriorSamples.get(vertex).asList());
    }

    @Test
    public void samplesContinuousPrior() {

        TensorBayesNet bayesNet = TensorMCMCTestDistributions.createSumOfGaussianDistribution(20.0, 1.0, 46.0);

        NetworkSamples posteriorSamples = TensorHamiltonian.getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            2000,
            20,
            0.1,
            random
        );

        Vertex<DoubleTensor> A = bayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = bayesNet.getContinuousLatentVertices().get(1);

        TensorMCMCTestDistributions.samplesMatchesSumOfGaussians(44.0, posteriorSamples.get(A).asList(), posteriorSamples.get(B).asList());
    }

    @Test
    public void samplesFromDonut() {

        TensorBayesNet donutBayesNet = TensorMCMCTestDistributions.create2DDonutDistribution();

        NetworkSamples samples = TensorHamiltonian.getPosteriorSamples(
            donutBayesNet,
            donutBayesNet.getLatentVertices(),
            2500,
            15,
            0.02,
            random
        );

        Vertex<DoubleTensor> A = donutBayesNet.getContinuousLatentVertices().get(0);
        Vertex<DoubleTensor> B = donutBayesNet.getContinuousLatentVertices().get(1);

        TensorMCMCTestDistributions.samplesMatch2DDonut(samples.get(A).asList(), samples.get(B).asList());
    }
}

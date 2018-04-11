package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class HamiltonianTest {

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplesGaussian() {
        GaussianVertex A = new GaussianVertex(0.0, 1, random);
        BayesNet bayesNet = new BayesNet(A.getConnectedGraph());

        NetworkSamples posteriorSamples = Hamiltonian.getPosteriorSamples(
                bayesNet,
                Arrays.asList(A),
                50000,
                10,
                0.1,
                Arrays.asList(0.02),
                random
        );

        OptionalDouble averagePosteriorA = posteriorSamples.get(A).asList().stream()
                .mapToDouble(sample -> sample)
                .average();

        assertEquals(0.0, averagePosteriorA.getAsDouble(), 0.1);
    }

    @Test
    public void samplesContinuousPrior() {
        DoubleVertex A = new GaussianVertex(20.0, 1.0, random);
        DoubleVertex B = new GaussianVertex(20.0, 1.0, random);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex C = new GaussianVertex(A.plus(B), new ConstantDoubleVertex(1.0), random);
        C.observe(46.0);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, C));
        bayesNet.probeForNonZeroMasterP(100);

        NetworkSamples posteriorSamples = Hamiltonian.getPosteriorSamples(
                bayesNet,
                Arrays.asList(A, B),
                25000,
                10,
                0.1,
                Arrays.asList(21.0, 22.5),
                random
        );

        OptionalDouble averagePosteriorA = posteriorSamples.get(A).asList().stream()
                .mapToDouble(sample -> sample)
                .average();

        OptionalDouble averagePosteriorB = posteriorSamples.get(B).asList().stream()
                .mapToDouble(sample -> sample)
                .average();

        assertEquals(44.0, averagePosteriorA.getAsDouble() + averagePosteriorB.getAsDouble(), 0.1);
    }

    @Test
    public void samplesFromDonut() {
        DoubleVertex A = new GaussianVertex(0, 1, random);
        DoubleVertex B = new GaussianVertex(0, 1, random);

        DoubleVertex D = new GaussianVertex((A.multiply(A)).plus(B.multiply(B)), 0.03, random);
        D.observe(0.5);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, D));

        NetworkSamples samples = Hamiltonian.getPosteriorSamples(
                bayesNet,
                Arrays.asList(A, B),
                25000,
                10,
                0.005,
                Arrays.asList(Math.sqrt(0.5), 0.0),
                random
        );

        List<Double> samplesA = samples.get(A).asList();
        List<Double> samplesB = samples.get(B).asList();

        boolean topOfDonut, rightOfDonut, bottomOfDonut, leftOfDonut, middleOfDonut;
        topOfDonut = rightOfDonut = bottomOfDonut = leftOfDonut = middleOfDonut = false;

        for (int i = 0; i < samplesA.size(); i++) {
            double sampleFromA = samplesA.get(i);
            double sampleFromB = samplesB.get(i);

            if (sampleFromA > -0.2 && sampleFromA < 0.2 && sampleFromB > 0.6 && sampleFromB < 0.8) {
                topOfDonut = true;
            } else if (sampleFromA > 0.6 && sampleFromA < 0.8 && sampleFromB > -0.2 && sampleFromB < 0.2) {
                rightOfDonut = true;
            } else if (sampleFromA > -0.2 && sampleFromA < 0.2 && sampleFromB > -0.8 && sampleFromB < -0.6) {
                bottomOfDonut = true;
            } else if (sampleFromA > -0.8 && sampleFromA < -0.6 && sampleFromB > -0.2 && sampleFromB < 0.2) {
                leftOfDonut = true;
            } else if (sampleFromA > 0.35 && sampleFromA < 0.45 && sampleFromB > 0.35 && sampleFromB < 0.45) {
                middleOfDonut = true;
            }
        }


        assertTrue(topOfDonut && rightOfDonut && bottomOfDonut && leftOfDonut && !middleOfDonut);
    }
}

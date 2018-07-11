package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.CategoricalVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class MetropolisHastingsTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesContinuousPrior() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            bayesNet,
            Arrays.asList(A, B),
            100000,
            random
        );

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages().scalar();

        double actual = averagePosteriorA + averagePosteriorB;
        assertEquals(44.0, actual, 0.1);
    }

    @Test
    public void samplesContinuousTensorPrior() {

        int[] shape = new int[]{1, 1};
        DoubleVertex A = new GaussianVertex(shape, 20.0, 1.0);
        DoubleVertex B = new GaussianVertex(shape, 20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);
        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            bayesNet,
            Arrays.asList(A, B),
            100000,
            random
        );

        DoubleTensor averagePosteriorA = posteriorSamples.getDoubleTensorSamples(A).getAverages();
        DoubleTensor averagePosteriorB = posteriorSamples.getDoubleTensorSamples(B).getAverages();

        DoubleTensor allActuals = averagePosteriorA.plus(averagePosteriorB);

        for (double actual : allActuals.asFlatDoubleArray()) {
            assertEquals(44.0, actual, 0.1);
        }
    }

    @Test
    public void samplesSimpleDiscretePrior() {

        CategoricalVertex A = new CategoricalVertex(0.5);

        DoubleVertex B = If.isTrue(A)
            .then(0.9)
            .orElse(0.1);

        CategoricalVertex C = new CategoricalVertex(B);

        C.observe(true);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, C));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            bayesNet,
            Collections.singletonList(A),
            10000,
            random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.9, postProbTrue, 0.01);
    }

    @Test
    public void samplesComplexDiscretePrior() {

        CategoricalVertex A = new CategoricalVertex(0.5);
        CategoricalVertex B = new CategoricalVertex(0.5);

        BoolVertex C = A.or(B);

        DoubleVertex D = If.isTrue(C)
            .then(0.9)
            .orElse(0.1);

        CategoricalVertex E = new CategoricalVertex(D);

        E.observe(true);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, C, D, E));
        bayesNet.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            bayesNet,
            Collections.singletonList(A),
            100000,
            random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.643, postProbTrue, 0.01);
    }

    @Test
    public void samplesFromPriorWithObservedDeterministic() {

        CategoricalVertex A = new CategoricalVertex(0.5);
        CategoricalVertex B = new CategoricalVertex(0.5);
        BoolVertex C = A.or(B);
        C.observe(false);

        BayesianNetwork net = new BayesianNetwork(A.getConnectedGraph());
        net.probeForNonZeroProbability(100, random);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            net,
            Collections.singletonList(A),
            10000,
            random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v.scalar());

        assertEquals(0.5, postProbTrue, 0.01);
    }

}

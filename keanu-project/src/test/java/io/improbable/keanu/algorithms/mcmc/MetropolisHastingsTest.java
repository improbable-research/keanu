package io.improbable.keanu.algorithms.mcmc;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collections;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class MetropolisHastingsTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesContinuousPrior() {

        DoubleVertex A = VertexOfType.gaussian(20.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(20.0, 1.0);

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = VertexOfType.gaussian(A.plus(B), ConstantVertex.of(1.0));

        Cobserved.observe(DoubleTensor.scalar(46.0));

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
        DoubleVertex A = new DistributionVertexBuilder()
            .shaped(shape)
            .withInput(ParameterName.MU, 20.0)
            .withInput(ParameterName.SIGMA, 1.0)
            .gaussian();
        DoubleVertex B = new DistributionVertexBuilder()
            .shaped(shape)
            .withInput(ParameterName.MU, 20.0)
            .withInput(ParameterName.SIGMA, 1.0)
            .gaussian();

        A.setValue(20.0);
        B.setValue(20.0);

        DoubleVertex Cobserved = VertexOfType.gaussian(A.plus(B), ConstantVertex.of(1.0));
        Cobserved.observe(DoubleTensor.scalar(46.0));

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

        Flip A = new Flip(0.5);

        DoubleVertex B = If.isTrue(A)
            .then(0.9)
            .orElse(0.1);

        Flip C = new Flip(B);

        C.observe(BooleanTensor.scalar(true));

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

        Flip A = new Flip(0.5);
        Flip B = new Flip(0.5);

        BooleanVertex C = A.or(B);

        DoubleVertex D = If.isTrue(C)
            .then(0.9)
            .orElse(0.1);

        Flip E = new Flip(D);

        E.observe(BooleanTensor.scalar(true));

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

        Flip A = new Flip(0.5);
        Flip B = new Flip(0.5);
        BooleanVertex C = A.or(B);
        C.observe(BooleanTensor.scalar(false));

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

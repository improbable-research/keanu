package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.OptionalDouble;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class MetropolisHastingsTest {

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplesContinuousPrior() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0, random);
        DoubleVertex B = new GaussianVertex(20.0, 1.0, random);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), new ConstantDoubleVertex(1.0), random);

        Cobserved.observe(46.0);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, Cobserved));
        bayesNet.probeForNonZeroMasterP(100);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
                bayesNet,
                Arrays.asList(A, B),
                100000,
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
    public void samplesSimpleDiscretePrior() {

        Flip A = new Flip(0.5, random);

        DoubleVertex B = new DoubleUnaryOpLambda<>(A, val -> val ? 0.9 : 0.1);

        Flip C = new Flip(B, random);

        C.observe(true);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, C));
        bayesNet.probeForNonZeroMasterP(100);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
                bayesNet,
                Collections.singletonList(A),
                10000,
                random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v);

        assertEquals(0.9, postProbTrue, 0.01);
    }

    @Test
    public void samplesComplexDiscretePrior() {

        Flip A = new Flip(0.5, random);
        Flip B = new Flip(0.5, random);

        BoolVertex C = A.or(B);
        DoubleVertex D = new DoubleUnaryOpLambda<>(C, val -> val ? 0.9 : 0.1);

        Flip E = new Flip(D, random);

        E.observe(true);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, C, D, E));
        bayesNet.probeForNonZeroMasterP(100);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
                bayesNet,
                Collections.singletonList(A),
                100000,
                random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v);

        assertEquals(0.643, postProbTrue, 0.01);
    }

    @Test
    public void samplesFromPriorWithObservedDeterministic() {

        Flip A = new Flip(0.5, random);
        Flip B = new Flip(0.5, random);
        BoolVertex C = A.or(B);
        C.observe(false);

        BayesNet net = new BayesNet(A.getConnectedGraph());
        net.probeForNonZeroMasterP(100);

        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
                net,
                Collections.singletonList(A),
                10000,
                random
        );

        double postProbTrue = posteriorSamples.get(A).probability(v -> v);

        assertEquals(0.5, postProbTrue, 0.01);
    }

}

package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.sampling.Prior;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.CastBoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static io.improbable.keanu.vertices.bool.BoolVertex.not;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

public class BoolVertexTest {

    private KeanuRandom random;
    private Flip v1;
    private Flip v2;
    private double pV2 = 0.4;
    private double pV1 = 0.25;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        v1 = new Flip(pV1);
        v2 = new Flip(pV2);
    }

    @Test
    public void doesOr() {
        BoolVertex v3 = v1.or(v2);

        v1.setValue(true);
        v2.setValue(false);

        assertTrue(v3.eval().scalar());
    }

    @Test
    public void doesAnd() {
        BoolVertex v3 = v1.and(v2);

        v1.setValue(true);
        v2.setValue(false);

        assertFalse(v3.eval().scalar());
    }

    @Test
    public void doesNot() {
        BoolVertex v3 = not(v1);

        v1.setValue(true);

        assertFalse(v3.eval().scalar());
    }

    @Test
    public void TheOperatorsAreExecutedInOrder() {
        Flip v3 = new Flip(0.5);

        BoolVertex v4 = v1.and(v2).or(v3); // (v1 AND v2) OR v3
        BoolVertex v5 = v1.and(v2.or(v3)); // v1 AND (v2 OR v3)

        v1.setValue(false);
        v2.setValue(true);
        v3.setValue(true);

        assertTrue(v4.eval().scalar());
        assertFalse(v5.eval().scalar());
    }

    @Test
    public void YouCanSpecifyYourOwnOrderingOfOperations() {
        Flip v3 = new Flip(0.5);
        BoolVertex v5 = v1.and(v2.or(v3));

        v1.setValue(false);
        v2.setValue(true);
        v3.setValue(true);
    }

    @Test
    public void youCanCombineTheOperatorsInDisjunctiveNormalForm() {
        assertFalse(xor(false, false));
        assertTrue(xor(false, true));
        assertTrue(xor(true, false));
        assertFalse(xor(true, true));
    }

    private boolean xor(boolean b1, boolean b2) {
        BoolVertex v3 =
            v1.and(not(v2))
            .or(not(v1).and(v2));
        v1.setValue(b1);
        v2.setValue(b2);
        return v3.eval().scalar();
    }

    @Test
    public void orProbabilityIsCorrect() {
        BoolVertex v3 = v1.or(v2);

        double pV3True = orProbability(pV1, pV2);

        assertEquals(priorProbabilityTrue(v3, 10000, random), pV3True, 0.01);
    }

    @Test
    public void andProbabilityIsCorrect() {
        BoolVertex v3 = v1.and(v2);

        double pV3True = andProbability(pV1, pV2);

        assertEquals(priorProbabilityTrue(v3, 10000, random), pV3True, 0.01);
    }

    @Test
    public void castVertexWorksAsExpected() {
        double p = 0.5;

        Flip f = new Flip(0.5);

        CastBoolVertex a = new CastBoolVertex(f);

        assertEquals(priorProbabilityTrue(a, 10000, random), p, 0.01);
    }

    @Test
    public void constantVertexWorksAsExpected() {
        double p = 0.5;

        Flip f = new Flip(0.5);
        ConstantBoolVertex tru = ConstantVertex.of(true);
        ConstantBoolVertex fal = ConstantVertex.of(false);

        BoolVertex a = f.and(tru).or(fal);

        assertEquals(priorProbabilityTrue(a, 10000, random), p, 0.01);
    }

    @Test
    public void canObserveArrayOfValues() {
        BoolVertex flip = new Flip(0.5);
        boolean[] observation = new boolean[]{true, false, true};
        flip.observe(observation);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetAndCascadeArrayOfValues() {
        BoolVertex flip = new Flip(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setAndCascade(values);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetValueArrayOfValues() {
        BoolVertex flip = new Flip(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setValue(values);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetValueAsScalarOnNonScalarVertex() {
        BoolVertex flip = new Flip(new int[]{2, 1}, 0.5);
        flip.setValue(true);
        assertArrayEquals(new Boolean[]{true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetAndCascadeAsScalarOnNonScalarVertex() {
        BoolVertex flip = new Flip(new int[]{2, 1}, 0.5);
        flip.setAndCascade(true);
        assertArrayEquals(new Boolean[]{true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canPluckValue() {
        BoolVertex flip = new Flip(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setAndCascade(values);
        assertEquals(true, flip.pluck(0, 0).getValue().scalar());
    }

    private double andProbability(double pA, double pB) {
        return pA * pB;
    }

    private double orProbability(double pA, double pB) {
        return pA + pB - (pA * pB);
    }

    public static double priorProbabilityTrue(Vertex<? extends Tensor<Boolean>> vertex, int sampleCount, KeanuRandom random) {
        BayesianNetwork net = new BayesianNetwork(vertex.getConnectedGraph());

        NetworkSamples samples = Prior.sample(net, Collections.singletonList(vertex), sampleCount, random);
        return samples.get(vertex).probability(val -> val.scalar());
    }

}

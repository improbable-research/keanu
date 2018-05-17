package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.sampling.Prior;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.CastBoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static io.improbable.keanu.vertices.bool.BoolVertex.If;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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

        assertTrue(v3.lazyEval());
    }

    @Test
    public void doesAnd() {
        BoolVertex v3 = v1.and(v2);

        v1.setValue(true);
        v2.setValue(false);

        assertTrue(!v3.lazyEval());
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
    public void ifProbabilityIsCorrect() {

        double pV3 = 0.1;
        Flip v3 = new Flip(pV3);

        Vertex<Boolean> v4 = If(v1, v2, v3);

        double pV4True = ifProbability(pV1, pV2, pV3);

        assertEquals(priorProbabilityTrue(v4, 10000, random), pV4True, 0.01);
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
        ConstantBoolVertex tru = new ConstantBoolVertex(true);
        ConstantBoolVertex fal = new ConstantBoolVertex(false);

        BoolVertex a = f.and(tru).or(fal);

        assertEquals(priorProbabilityTrue(a, 10000, random), p, 0.01);
    }


    private double andProbability(double pA, double pB) {
        return pA * pB;
    }

    private double orProbability(double pA, double pB) {
        return pA + pB - (pA * pB);
    }

    private double ifProbability(double pThn, double pThnIsValue, double pElsIsValue) {
        double pThnAndThnIsValue = pThn * pThnIsValue;
        double pElsAndElsIsValue = (1 - pThn) * pElsIsValue;

        return pThnAndThnIsValue + pElsAndElsIsValue;
    }

    public static double priorProbabilityTrue(Vertex<Boolean> vertex, int sampleCount, KeanuRandom random) {
        BayesNet net = new BayesNet(vertex.getConnectedGraph());

        NetworkSamples samples = Prior.sample(net, Collections.singletonList(vertex), sampleCount, random);
        return samples.get(vertex).probability(val -> val);
    }

}

package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.algorithms.mcmc.KeanuMetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.CastToBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static io.improbable.keanu.vertices.bool.BooleanVertex.not;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class BooleanVertexTest {

    private KeanuRandom random;
    private BernoulliVertex v1;
    private BernoulliVertex v2;
    private double pV2 = 0.4;
    private double pV1 = 0.25;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        v1 = new BernoulliVertex(pV1);
        v2 = new BernoulliVertex(pV2);
    }

    @Test
    public void doesOr() {
        BooleanVertex v3 = v1.or(v2);

        v1.setValue(true);
        v2.setValue(false);

        assertTrue(v3.eval().scalar());
    }

    @Test
    public void doesAnd() {
        BooleanVertex v3 = v1.and(v2);

        v1.setValue(true);
        v2.setValue(false);

        assertFalse(v3.eval().scalar());
    }

    @Test
    public void doesNot() {
        BooleanVertex v3 = not(v1);

        v1.setValue(true);

        assertFalse(v3.eval().scalar());
    }

    @Test
    public void doesEqualTo() {
        v1.setValue(true);
        v2.setValue(false);
        BooleanVertex v3 = ConstantVertex.of(true);

        assertFalse(v1.equalTo(v2).eval().scalar());
        assertTrue(v1.notEqualTo(v2).eval().scalar());
        assertFalse(v2.equalTo(v3).eval().scalar());
        assertTrue(v2.notEqualTo(v3).eval().scalar());
    }

    @Test
    public void TheOperatorsAreExecutedInOrder() {
        BernoulliVertex v3 = new BernoulliVertex(0.5);

        BooleanVertex v4 = v1.and(v2).or(v3); // (v1 AND v2) OR v3
        BooleanVertex v5 = v1.and(v2.or(v3)); // v1 AND (v2 OR v3)

        v1.setValue(false);
        v2.setValue(true);
        v3.setValue(true);

        assertTrue(v4.eval().scalar());
        assertFalse(v5.eval().scalar());
    }

    @Test
    public void canSpecifyYourOwnOrderingOfOperations() {
        BernoulliVertex v3 = new BernoulliVertex(0.5);

        v1.setValue(false);
        v2.setValue(true);
        v3.setValue(true);
    }

    @Test
    public void canCombineTheOperatorsInDisjunctiveNormalForm() {
        assertFalse(xor(false, false));
        assertTrue(xor(false, true));
        assertTrue(xor(true, false));
        assertFalse(xor(true, true));
    }

    private boolean xor(boolean b1, boolean b2) {
        BooleanVertex v3 =
            v1.and(not(v2))
                .or(not(v1).and(v2));
        v1.setValue(b1);
        v2.setValue(b2);
        return v3.eval().scalar();
    }

    @Test
    public void orProbabilityIsCorrect() {
        BooleanVertex v3 = v1.or(v2);

        double pV3True = orProbability(pV1, pV2);

        assertEquals(priorProbabilityTrue(v3, 10000, random), pV3True, 0.01);
    }

    @Test
    public void andProbabilityIsCorrect() {
        BooleanVertex v3 = v1.and(v2);

        double pV3True = andProbability(pV1, pV2);

        assertEquals(priorProbabilityTrue(v3, 10000, random), pV3True, 0.01);
    }

    @Test
    public void castVertexWorksAsExpected() {
        double p = 0.5;

        BernoulliVertex f = new BernoulliVertex(0.5);

        CastToBooleanVertex a = new CastToBooleanVertex(f);

        assertEquals(priorProbabilityTrue(a, 10000, random), p, 0.01);
    }

    @Test
    public void constantVertexWorksAsExpected() {
        double p = 0.5;

        BernoulliVertex f = new BernoulliVertex(0.5);
        ConstantBooleanVertex tru = ConstantVertex.of(true);
        ConstantBooleanVertex fal = ConstantVertex.of(false);

        BooleanVertex a = f.and(tru).or(fal);

        assertEquals(priorProbabilityTrue(a, 10000, random), p, 0.01);
    }

    @Test
    public void canObserveArrayOfValues() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        boolean[] observation = new boolean[]{true, false, true};
        flip.observe(observation);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canObserveTensor() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        BooleanTensor observation = BooleanTensor.create(new boolean[]{true, false, true, false}, new long[]{2, 2});
        flip.observe(observation);
        assertArrayEquals(observation.asFlatArray(), flip.getValue().asFlatArray());
        assertArrayEquals(flip.getShape(), observation.getShape());
    }

    @Test
    public void canSetAndCascadeArrayOfValues() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setAndCascade(values);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetValueArrayOfValues() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setValue(values);
        assertArrayEquals(new Boolean[]{true, false, true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetValueAsScalarOnNonScalarVertex() {
        BooleanVertex flip = new BernoulliVertex(new long[]{2, 1}, 0.5);
        flip.setValue(true);
        assertArrayEquals(new Boolean[]{true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canSetAndCascadeAsScalarOnNonScalarVertex() {
        BooleanVertex flip = new BernoulliVertex(new long[]{2, 1}, 0.5);
        flip.setAndCascade(true);
        assertArrayEquals(new Boolean[]{true}, flip.getValue().asFlatArray());
    }

    @Test
    public void canPluckValue() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        boolean[] values = new boolean[]{true, false, true};
        flip.setAndCascade(values);
        assertEquals(true, flip.take(0).getValue().scalar());
    }

    @Test
    public void canReshape() {
        BooleanVertex flip = new BernoulliVertex(0.5);
        flip.setAndCascade(BooleanTensor.trues(2, 2));
        assertArrayEquals(flip.getShape(), new long[]{2, 2});
        BooleanVertex reshaped = flip.reshape(4, 1);
        assertArrayEquals(reshaped.getShape(), new long[]{4, 1});
    }

    @Test
    public void canConcat() {
        BooleanVertex A = new BernoulliVertex(0.5);
        A.setValue(BooleanTensor.trues(2, 2));

        BooleanVertex B = new BernoulliVertex(0.5);
        B.setValue(BooleanTensor.falses(2, 2));

        BooleanVertex concatDimZero = BooleanVertex.concat(0, A, A);
        assertArrayEquals(concatDimZero.getShape(), new long[]{4, 2});

        BooleanVertex concatDimOne = BooleanVertex.concat(1, A, B);
        assertArrayEquals(concatDimOne.getShape(), new long[]{2, 4});
    }

    private double andProbability(double pA, double pB) {
        return pA * pB;
    }

    private double orProbability(double pA, double pB) {
        return pA + pB - (pA * pB);
    }

    public static double priorProbabilityTrue(Vertex<? extends Tensor<Boolean>> vertex, int sampleCount, KeanuRandom random) {
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(vertex.getConnectedGraph());

        long trueCount = KeanuMetropolisHastings.withDefaultConfigFor(model, random)
            .generatePosteriorSamples(model, Collections.singletonList(vertex)).stream()
            .limit(sampleCount)
            .filter(state -> state.get(vertex).scalar())
            .count();

        return trueCount / (double) sampleCount;
    }

}

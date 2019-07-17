package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.BooleanVertexTest;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;

import java.util.Collections;
import java.util.stream.Collectors;

import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex.FALSE;
import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class IfVertexTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void canExtractPartialFromTruePredicate() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, true, true}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        PartialsOf dC = Differentiator.reverseModeAutoDiff(c, a, b);
        DoubleTensor dCda = dC.withRespectTo(a);
        DoubleTensor dCdb = dC.withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        PartialsOf dIfVertex = Differentiator.reverseModeAutoDiff(ifVertex, a, b);
        DoubleTensor dIfdA = dIfVertex.withRespectTo(a);
        DoubleTensor dIfdB = dIfVertex.withRespectTo(b);

        Assert.assertArrayEquals(dCda.asFlatDoubleArray(), dIfdA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(dCdb.asFlatDoubleArray(), dIfdB.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dCdb.getShape(), dIfdB.getShape());
    }

    @Test
    public void canExtractPartialFromTruePredicateDifferentRankOf() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, true, true, false, true, true, true, false}, 2, 2, 2));

        UniformVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.scalar(5.0));

        UniformVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = b.div(a);

        PartialsOf dC = Differentiator.reverseModeAutoDiff(c, a, b);
        DoubleTensor dCda = dC.withRespectTo(a);
        DoubleTensor dCdb = dC.withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);


        DoubleTensor dIfdAForward = Differentiator.forwardModeAutoDiff(a, ifVertex).of(ifVertex);
        DoubleTensor dIfdBForward = Differentiator.forwardModeAutoDiff(b, ifVertex).of(ifVertex);

        DoubleTensor dIfdAReverse = Differentiator.reverseModeAutoDiff(ifVertex, a).withRespectTo(a);
        DoubleTensor dIfdBReverse = Differentiator.reverseModeAutoDiff(ifVertex, b).withRespectTo(b);

        Assert.assertArrayEquals(dCda.getShape(), dIfdAForward.getShape());
        Assert.assertArrayEquals(dCdb.getShape(), dIfdBForward.getShape());

        Assert.assertArrayEquals(dIfdAForward.asFlatDoubleArray(), dIfdAReverse.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(dIfdBForward.asFlatDoubleArray(), dIfdBReverse.asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canExtractPartialFromFalsePredicate() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{false, false, false, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        PartialsOf dD = Differentiator.reverseModeAutoDiff(d, a, b);
        DoubleTensor dDda = dD.withRespectTo(a);
        DoubleTensor dDdb = dD.withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        PartialsOf dIfVertex = Differentiator.reverseModeAutoDiff(ifVertex, a, b);
        DoubleTensor dIfdA = dIfVertex.withRespectTo(a);
        DoubleTensor dIfdB = dIfVertex.withRespectTo(b);

        Assert.assertArrayEquals(dDda.asFlatDoubleArray(), dIfdA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(dDdb.asFlatDoubleArray(), dIfdB.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dDda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dDdb.getShape(), dIfdB.getShape());
    }

    @Test
    public void canExtractPartialFromMixedPredicate() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dCda = Differentiator.reverseModeAutoDiff(c, a).withRespectTo(a);
        DoubleTensor dDda = Differentiator.reverseModeAutoDiff(d, a).withRespectTo(a);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        DoubleTensor dIfdA = Differentiator.reverseModeAutoDiff(ifVertex, a).withRespectTo(a);

        Assert.assertArrayEquals(new double[]{
            5, 7,
            0, 0,
            0, 5,
            0, 6,
            0, 0,
            5, 7,
            0, 7,
            0, 8
        }, dIfdA.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dDda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
    }

    @Test
    public void canExtractPartialFromMixedPredicateWithDifferentParentsAndFillInZeroes() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.create(new double[]{9, 10, 11, 12}, 2, 2));

        DoubleVertex e = new UniformVertex(0, 10);
        e.setValue(DoubleTensor.create(new double[]{13, 14, 15, 16}, 2, 2));

        DoubleVertex f = d.matrixMultiply(e);

        DoubleTensor dCda = Differentiator.reverseModeAutoDiff(c, a).withRespectTo(a);
        DoubleTensor dFdd = Differentiator.reverseModeAutoDiff(f, d).withRespectTo(d);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(f);

        PartialsOf dIfVertex = Differentiator.reverseModeAutoDiff(ifVertex, a, d);
        DoubleTensor dIfdA = dIfVertex.withRespectTo(a);
        DoubleTensor dIfdD = dIfVertex.withRespectTo(d);

        Assert.assertArrayEquals(new double[]{
            5, 7,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            5, 7,
            0, 0,
            0, 0
        }, dIfdA.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(new double[]{
            0, 0,
            0, 0,
            14, 16,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            14, 16
        }, dIfdD.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dFdd.getShape(), dIfdD.getShape());
    }

    @Test
    public void canExtractPartialFromMixedPredicateWithDifferentParentsRankThree() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, false, true, false, true, false, true, false}, 2, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(10, 18).reshape(2, 2, 2));

        DoubleVertex c = a.times(b);

        DoubleVertex d = new UniformVertex(0, 10);
        d.setValue(DoubleTensor.arange(20, 28).reshape(2, 2, 2));

        DoubleVertex e = new UniformVertex(0, 10);
        e.setValue(DoubleTensor.arange(30, 38).reshape(2, 2, 2));

        DoubleVertex f = d.plus(e);

        DoubleTensor dCda = Differentiator.reverseModeAutoDiff(c, a).withRespectTo(a);
        DoubleTensor dFdd = Differentiator.reverseModeAutoDiff(f, d).withRespectTo(d);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(f);

        PartialsOf dIfVertex = Differentiator.reverseModeAutoDiff(ifVertex, a, d);
        DoubleTensor dIfdA = dIfVertex.withRespectTo(a);
        DoubleTensor dIfdD = dIfVertex.withRespectTo(d);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dFdd.getShape(), dIfdD.getShape());
    }

    @Test
    public void canExtractValueFromMixedPredicate() {
        BooleanVertex bool = new ConstantBooleanVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        Assert.assertArrayEquals(new double[]{
            19, 34,
            43, 46
        }, ifVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

    @Test
    public void canRunGradientOptimiserThroughIfWhenTrue() {
        UniformVertex a = new UniformVertex(2, 2.5);
        a.setValue(2.1);
        UniformVertex b = new UniformVertex(1, 1.5);
        b.setValue(1.1);

        BooleanVertex leftFlip = new BernoulliVertex(0.5);
        BooleanVertex rightFlip = new BernoulliVertex(0.5);
        leftFlip.observe(false);
        rightFlip.observe(true);

        DoubleVertex ifVertex = If.isTrue(leftFlip.or(rightFlip))
            .then(a)
            .orElse(b);

        double sigma = 2.0;

        GaussianVertex observedIf = new GaussianVertex(ifVertex, sigma);

        observedIf.observe(2.25);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(observedIf.getConnectedGraph());
        GradientOptimizer gradientOptimizer = Keanu.Optimizer.Gradient.of(bayesianNetwork);

        gradientOptimizer.maxLikelihood();

        Assert.assertEquals(2.25, a.getValue().scalar(), 1e-6);
    }

    @Test
    public void canRunGradientOptimiserThroughIfWhenFalse() {
        UniformVertex a = new UniformVertex(2, 2.5);
        a.setValue(2.1);
        UniformVertex b = new UniformVertex(1, 1.5);
        b.setValue(1.1);

        BooleanVertex leftFlip = new BernoulliVertex(0.5);
        BooleanVertex rightFlip = new BernoulliVertex(0.5);
        leftFlip.observe(false);
        rightFlip.observe(false);

        DoubleVertex ifVertex = If.isTrue(leftFlip.or(rightFlip))
            .then(a)
            .orElse(b);

        double sigma = 2.0;

        GaussianVertex observedIf = new GaussianVertex(ifVertex, sigma);

        observedIf.observe(1.25);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(observedIf.getConnectedGraph());
        GradientOptimizer gradientOptimizer = Keanu.Optimizer.Gradient.of(bayesianNetwork);

        gradientOptimizer.maxLikelihood();

        Assert.assertEquals(1.25, b.getValue().scalar(), 1e-6);
    }

    @Test
    public void correctBranchTaken() {

        BooleanVertex predicate = new BernoulliVertex(0.5);

        IntegerVertex ifIsTrue = If.isTrue(predicate)
            .then(1)
            .orElse(0);

        predicate.setAndCascade(true);
        assertEquals(1, ifIsTrue.getValue().scalar().intValue());

        predicate.setAndCascade(false);
        assertEquals(0, ifIsTrue.getValue().scalar().intValue());
    }

    @Test
    public void expectedValueMatchesBranchProbability() {

        double p = 0.5;
        int thenValue = 2;
        int elseValue = 4;
        double expectedMean = p * thenValue + (1 - p) * elseValue;

        BernoulliVertex predicate = new BernoulliVertex(p);

        IntegerVertex vertex = If.isTrue(predicate)
            .then(thenValue)
            .orElse(elseValue);

        assertEquals(calculateMeanOfVertex(vertex), expectedMean, 0.01);
    }

    private static double calculateMeanOfVertex(IntegerVertex vertex) {
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(vertex.getConnectedGraph());

        return Keanu.Sampling.MetropolisHastings.withDefaultConfig(KeanuRandom.getDefaultRandom())
            .generatePosteriorSamples(model, Collections.singletonList(vertex)).stream()
            .limit(2000)
            .collect(Collectors.averagingInt((NetworkSample state) -> state.get(vertex).scalar()));
    }

    @Test
    public void functionsAsIf() {

        BooleanVertex predicate = new BernoulliVertex(0.5);

        BooleanVertex ifIsTrue = If.isTrue(predicate)
            .then(TRUE)
            .orElse(FALSE);

        predicate.setAndCascade(true);
        assertTrue(ifIsTrue.getValue().scalar());

        predicate.setAndCascade(false);
        assertFalse(ifIsTrue.getValue().scalar());
    }

    @Test
    public void ifProbabilityIsCorrect() {

        double pV2 = 0.4;
        double pV1 = 0.25;
        BernoulliVertex v1 = new BernoulliVertex(pV1);
        BernoulliVertex v2 = new BernoulliVertex(pV2);

        double pV3 = 0.1;
        BernoulliVertex v3 = new BernoulliVertex(pV3);

        BooleanVertex v4 = If.isTrue(v1)
            .then(v2)
            .orElse(v3);

        double pV4True = ifProbability(pV1, pV2, pV3);

        assertEquals(BooleanVertexTest.priorProbabilityTrue(v4, 10000, KeanuRandom.getDefaultRandom()), pV4True, 0.01);
    }

    private double ifProbability(double pThn, double pThnIsValue, double pElsIsValue) {
        double pThnAndThnIsValue = pThn * pThnIsValue;
        double pElsAndElsIsValue = (1 - pThn) * pElsIsValue;

        return pThnAndThnIsValue + pElsAndElsIsValue;
    }

}

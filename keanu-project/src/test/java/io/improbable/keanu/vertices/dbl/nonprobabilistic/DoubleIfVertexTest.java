package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.dbl.Differentiator;
import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class DoubleIfVertexTest {

    @Test
    public void canExtractDualNumberFromTruePredicate() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, true, true, true}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dCdb = c.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dIfdB = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(b);

        Assert.assertArrayEquals(dCda.asFlatDoubleArray(), dIfdA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(dCdb.asFlatDoubleArray(), dIfdB.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dCdb.getShape(), dIfdB.getShape());
    }

    @Test
    public void canExtractDualNumberFromTruePredicateDifferentRankOf() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, true, true, false, true, true, true, false}, 2, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.scalar(5.0));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.arange(0, 8).reshape(2, 2, 2));

        DoubleVertex c = a.times(b);
        DoubleVertex d = b.div(a);

        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dCdb = c.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dIfdB = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(b);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dCdb.getShape(), dIfdB.getShape());
    }

    @Test
    public void canExtractDualNumberFromFalsePredicate() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{false, false, false, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dDda = d.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dDdb = d.getDualNumber().getPartialDerivatives().withRespectTo(b);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dIfdB = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(b);

        Assert.assertArrayEquals(dDda.asFlatDoubleArray(), dIfdA.asFlatDoubleArray(), 1e-6);
        Assert.assertArrayEquals(dDdb.asFlatDoubleArray(), dIfdB.asFlatDoubleArray(), 1e-6);

        Assert.assertArrayEquals(dDda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dDdb.getShape(), dIfdB.getShape());
    }

    @Test
    public void canExtractDualNumberFromMixedPredicate() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex b = new UniformVertex(0, 10);
        b.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex c = a.matrixMultiply(b);
        DoubleVertex d = b.matrixMultiply(a);

        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dDda = d.getDualNumber().getPartialDerivatives().withRespectTo(a);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(d);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);

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
    public void canExtractDualNumberFromMixedPredicateWithDifferentParentsAndFillInZeroes() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

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

        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dFdd = f.getDualNumber().getPartialDerivatives().withRespectTo(d);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(f);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dIfdD = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(d);

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
    public void canExtractDualNumberFromMixedPredicateWithDifferentParentsRankThree() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, false, true, false, true, false, true, false}, 2, 2, 2));

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

        DoubleTensor dCda = c.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dFdd = f.getDualNumber().getPartialDerivatives().withRespectTo(d);

        DoubleVertex ifVertex = If.isTrue(bool)
            .then(c)
            .orElse(f);

        DoubleTensor dIfdA = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(a);
        DoubleTensor dIfdD = ifVertex.getDualNumber().getPartialDerivatives().withRespectTo(d);

        Assert.assertArrayEquals(dCda.getShape(), dIfdA.getShape());
        Assert.assertArrayEquals(dFdd.getShape(), dIfdD.getShape());
    }

    @Test
    public void canExtractValueFromMixedPredicate() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

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

        BoolVertex leftFlip = new BernoulliVertex(0.5);
        BoolVertex rightFlip = new BernoulliVertex(0.5);
        leftFlip.observe(false);
        rightFlip.observe(true);

        DoubleVertex ifVertex = If.isTrue(leftFlip.or(rightFlip))
            .then(a)
            .orElse(b);

        double sigma = 2.0;

        GaussianVertex observedIf = new GaussianVertex(ifVertex, sigma);

        observedIf.observe(2.25);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(observedIf.getConnectedGraph());
        GradientOptimizer gradientOptimizer = GradientOptimizer.of(bayesianNetwork);

        gradientOptimizer.maxLikelihood();

        Assert.assertEquals(2.25, a.getValue().scalar(), 1e-6);
    }

    @Test
    public void canRunGradientOptimiserThroughIfWhenFalse() {
        UniformVertex a = new UniformVertex(2, 2.5);
        a.setValue(2.1);
        UniformVertex b = new UniformVertex(1, 1.5);
        b.setValue(1.1);

        BoolVertex leftFlip = new BernoulliVertex(0.5);
        BoolVertex rightFlip = new BernoulliVertex(0.5);
        leftFlip.observe(false);
        rightFlip.observe(false);

        DoubleVertex ifVertex = If.isTrue(leftFlip.or(rightFlip))
            .then(a)
            .orElse(b);

        double sigma = 2.0;

        GaussianVertex observedIf = new GaussianVertex(ifVertex, sigma);

        observedIf.observe(1.25);

        BayesianNetwork bayesianNetwork = new BayesianNetwork(observedIf.getConnectedGraph());
        GradientOptimizer gradientOptimizer = GradientOptimizer.of(bayesianNetwork);

        gradientOptimizer.maxLikelihood();

        Assert.assertEquals(1.25, b.getValue().scalar(), 1e-6);
    }

}

package io.improbable.keanu.vertices;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.Optimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.Arrays;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.IsInstanceOf.instanceOf;
import static org.junit.Assert.assertEquals;


public class AssertVertexTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void assertThrowsOnFalseConstBool() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        constBool.assertTrue().eval();
    }

    @Test
    public void assertPassesOnTrueConstBool() {
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(true));
        constBool.assertTrue().eval();
    }

    @Test
    public void lazyEvalThrowsOnFalseConstBool() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        constBool.assertTrue().lazyEval();
    }

    @Test
    public void assertPassesOnRVWithTruePredicate() {
        UniformVertex uniform = new UniformVertex(0, 5);
        BoolVertex predicate = uniform.lessThan(new ConstantDoubleVertex(new double[]{10}));
        predicate.assertTrue().eval();
    }

    @Test
    public void assertPassesOnRVWithFalsePredicate() {
        thrown.expect(AssertionError.class);
        UniformVertex uniform = new UniformVertex(0, 5);
        BoolVertex predicate = uniform.greaterThan(new ConstantDoubleVertex(new double[]{10}));
        predicate.assertTrue().eval();
    }

    @Test
    public void samplingWithAssertionWorks() {
        thrown.expect(AssertionError.class);
        GaussianVertex gaussian = new GaussianVertex(5, 1);
        gaussian.greaterThan(new ConstantDoubleVertex(1000)).assertTrue();

        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        MetropolisHastings.withDefaultConfig().generatePosteriorSamples(bayesianNetwork, bayesianNetwork.getLatentVertices()).generate(10);
    }

    @Test
    public void valuePropagationPassesOnTrueAssertion() {
        ConstantDoubleVertex constDouble = new ConstantDoubleVertex(3);
        constDouble.lessThan(new ConstantDoubleVertex(10)).assertTrue();
        constDouble.setAndCascade(8);
    }

    @Test
    public void valuePropagationThrowsOnFalseAssertion() {
        thrown.expect(AssertionError.class);
        ConstantDoubleVertex constDouble = new ConstantDoubleVertex(3);
        constDouble.lessThan(new ConstantDoubleVertex(10)).assertTrue();
        constDouble.setAndCascade(15);
    }

    @Test
    public void optimizerWithAssertionWorks() {
        thrown.expect(AssertionError.class);
        UniformVertex temperature = new UniformVertex(20, 30);
        GaussianVertex observedTemp = new GaussianVertex(temperature, 1);
        observedTemp.observe(29);
        temperature.greaterThan(new ConstantDoubleVertex(34)).assertTrue();
        KeanuOptimizer.of(temperature.getConnectedGraph()).maxAPosteriori();
    }

    @Test
    public void assertGivesCorrectErrorWhenLabelledAndMessagePresent() {
        thrown.expect(AssertionError.class);
        thrown.expectMessage("AssertVertex (testAssert): this is wrong");
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = constBool.assertTrue("this is wrong");
        assertVertex.setLabel("testAssert");
        assertVertex.eval();
    }

    @Test
    public void assertGivesCorrectErrorWhenLabelled() {
        thrown.expect(AssertionError.class);
        thrown.expectMessage("AssertVertex (testAssert)");
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = constBool.assertTrue();
        assertVertex.setLabel("testAssert");
        assertVertex.eval();
    }

    @Test
    public void assertGivesCorrectErrorWhenMessagePresent() {
        thrown.expect(AssertionError.class);
        thrown.expectMessage("AssertVertex: this is wrong");
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = constBool.assertTrue("this is wrong");
        assertVertex.eval();
    }

    @Test
    public void assertGivesCorrectErrorWhenPlain() {
        thrown.expect(AssertionError.class);
        thrown.expectMessage("AssertVertex");
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = constBool.assertTrue();
        assertVertex.eval();
    }

    @Test
    public void samplingWithAssertThatShouldntFire() {
        UniformVertex temperature = new UniformVertex(20., 30.);
        temperature.lessThan(new ConstantDoubleVertex(30)).assertTrue();
        temperature.greaterThan(new ConstantDoubleVertex(20)).assertTrue();

        GaussianVertex firstThermometer = new GaussianVertex(temperature, 2.5);
        GaussianVertex secondThermometer = new GaussianVertex(temperature, 5.);
        firstThermometer.observe(25.);
        secondThermometer.observe(30.);

        BayesianNetwork bayesNet = new BayesianNetwork(temperature.getConnectedGraph());
        MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            bayesNet,
            bayesNet.getLatentVertices(),
            100
        );
    }

    @Test
    public void optimizerWithAssertThatShoudlntFire() {
        UniformVertex temperature = new UniformVertex(20., 30.);
        temperature.lessThan(new ConstantDoubleVertex(35)).assertTrue();
        temperature.greaterThan(new ConstantDoubleVertex(15)).assertTrue();

        GaussianVertex firstThermometer = new GaussianVertex(temperature, 2.5);
        GaussianVertex secondThermometer = new GaussianVertex(temperature, 5.);
        firstThermometer.observe(25.);
        secondThermometer.observe(30.);

        BayesianNetwork bayesNet = new BayesianNetwork(temperature.getConnectedGraph());
        KeanuOptimizer.of(bayesNet).maxAPosteriori();
        assertEquals(26, temperature.getValue().scalar(), 0.1);
    }

    @Test
    public void canUseGradientOptimizerWithAssertVertex() {
        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);
        A.setValue(21.5);
        B.setAndCascade(21.5);

        A.greaterThan(new ConstantDoubleVertex(20)).assertTrue();
        B.greaterThan(new ConstantDoubleVertex(20)).assertTrue();

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);
        Cobserved.observe(46.0);

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
        Optimizer optimizer = KeanuOptimizer.of(bayesNet);
        assertThat(optimizer, instanceOf(GradientOptimizer.class));

        optimizer.maxAPosteriori();
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }
}

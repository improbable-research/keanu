package io.improbable.keanu.vertices;

import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class AssertVertexTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void assertThrowsOnFalseConstBool() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = new AssertVertex(constBool);
        assertVertex.eval();
    }

    @Test
    public void assertPassesOnTrueConstBool() {
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(true));
        AssertVertex assertVertex = new AssertVertex(constBool);
        assertVertex.eval();
    }

    @Test
    public void lazyEvalThrowsOFalseConstBool() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = new AssertVertex(constBool);
        assertVertex.lazyEval();
    }

    @Test
    public void assertPassesOnRVWithTruePredicate() {
        UniformVertex uniform = new UniformVertex(0,5);
        BoolVertex predicate = uniform.lessThan(new ConstantDoubleVertex(new double[]{10}));
        AssertVertex assertVertex = new AssertVertex(predicate);
        assertVertex.eval();
    }

    @Test
    public void assertPassesOnRVWithFalsePredicate() {
        thrown.expect(AssertionError.class);
        UniformVertex uniform = new UniformVertex(0,5);
        BoolVertex predicate = uniform.greaterThan(new ConstantDoubleVertex(new double[]{10}));
        AssertVertex assertVertex = new AssertVertex(predicate);
        assertVertex.eval();
    }

    @Test
    public void samplingWithAssertionWorks() {
        thrown.expect(AssertionError.class);
        GaussianVertex uniform = new GaussianVertex(5,1);
        AssertVertex assertion = new AssertVertex(uniform.greaterThan(new ConstantDoubleVertex(10000)));
        GaussianVertex observingVertex = new GaussianVertex(uniform, 1);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(observingVertex.getConnectedGraph());
        MetropolisHastings.withDefaultConfig().generatePosteriorSamples(bayesianNetwork,bayesianNetwork.getLatentVertices()).generate(10);
    }

    @Test
    public void assertGivesCorrectErrorMessage() {
        thrown.expect(AssertionError.class);
        thrown.expectMessage("AssertVertex (testAssert): this is wrong");
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = new AssertVertex(constBool, "this is wrong");
        assertVertex.setLabel("testAssert");
        assertVertex.eval();
    }

}

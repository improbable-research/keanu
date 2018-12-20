package io.improbable.keanu.vertices;

import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
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
    public void assertThrowsOnConstBoolWithIncorrectExpected() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(true));
        AssertVertex assertVertex = new AssertVertex(constBool, BooleanTensor.create(false));
        assertVertex.eval();
    }

    @Test
    public void assertPassesOnConstBoolWithCorrectExpected() {
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(true));
        AssertVertex assertVertex = new AssertVertex(constBool, BooleanTensor.create(true));
        assertVertex.eval();
    }

    @Test
    public void lazyEvalThrowsOnConstBoolWithIncorrectExpected() {
        thrown.expect(AssertionError.class);
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(false));
        AssertVertex assertVertex = new AssertVertex(constBool, BooleanTensor.create(true));
        assertVertex.lazyEval();
    }

    @Test
    public void checkUsingPredicatesWorks() {
        UniformVertex uniform = new UniformVertex(Tensor.SCALAR_SHAPE,0,5);
        BoolVertex predicate = uniform.lessThan(new ConstantDoubleVertex(new double[]{10}, Tensor.SCALAR_SHAPE));
        AssertVertex assertVertex = new AssertVertex(predicate, BooleanTensor.create(true, Tensor.SCALAR_SHAPE));
        assertVertex.eval();
    }

    @Test
    public void checkSamplingWithAssertionWorks() {
        thrown.expect(AssertionError.class);
        GaussianVertex uniform = new GaussianVertex(5,1);
        AssertVertex assertion = new AssertVertex(uniform.greaterThan(new ConstantDoubleVertex(10000)), BooleanTensor.trues(Tensor.SCALAR_SHAPE));
        GaussianVertex observingVertex = new GaussianVertex(uniform, 1);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(observingVertex.getConnectedGraph());
        MetropolisHastings.withDefaultConfig().generatePosteriorSamples(bayesianNetwork,bayesianNetwork.getLatentVertices()).generate(1000);
    }

}

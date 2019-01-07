package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.distributions.gradient.Gaussian;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Test;
import org.omg.CORBA.BAD_CONTEXT;

import static org.junit.Assert.assertEquals;


public class DifferentiablePathCheckerTest {

    @Test
    public void testBasicDiffPath() {
        GaussianVertex a = new GaussianVertex(5., 4.);
        GaussianVertex b = new GaussianVertex(a, 1.);
        GaussianVertex c = new GaussianVertex(a, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(c.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void testBasicDiffFail() {
        GaussianVertex latentBeforeNonDiffOp = new GaussianVertex(5., 3.);
        DoubleVertex floorVertex = new FloorVertex(latentBeforeNonDiffOp);
        GaussianVertex latentAfterNonDiffOp = new GaussianVertex(floorVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(floorVertex.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void IfWithNonConstantPredicateIsntDiffable() {
        BernoulliVertex predicate = new BernoulliVertex(0.5);
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        GaussianVertex gaussianB = new GaussianVertex(5., 1.);
        DoubleVertex ifResult = If.isTrue(predicate).then(gaussianA).orElse(gaussianB);
        GaussianVertex postNonDiffVertex = new GaussianVertex(ifResult, 5);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(postNonDiffVertex.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void discreteLatentsDontHaveDiffablePath() {
        GaussianVertex mu = new GaussianVertex(5., 1.);
        PoissonVertex poisson = new PoissonVertex(mu);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(poisson.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void latentAsParentOfNonDiff() {
        GaussianVertex latent = new GaussianVertex(5., 1.);
        DoubleVertex nonDiffableVertex = new FloorVertex(latent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void observedLatentAsParentOfNonDiff() {
        GaussianVertex observedLatent = new GaussianVertex(5., 1.);
        observedLatent.observe(4);
        DoubleVertex nonDiffableVertex = new FloorVertex(observedLatent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void constantAsParentOfNonDiff() {
        DoubleVertex constantVertex = new ConstantDoubleVertex(20.);
        DoubleVertex nonDiffableVertex = new FloorVertex(constantVertex);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiablePathChecker.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

}

package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;


public class DifferentiableCheckerTest {

    @Test
    public void simpleDiffable() {
        GaussianVertex a = new GaussianVertex(5., 4.);
        GaussianVertex b = new GaussianVertex(a, 1.);
        GaussianVertex c = new GaussianVertex(a, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(c.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void simpleNonDiffable() {
        GaussianVertex latentBeforeNonDiffOp = new GaussianVertex(5., 3.);
        DoubleVertex floorVertex = new FloorVertex(latentBeforeNonDiffOp);
        GaussianVertex latentAfterNonDiffOp = new GaussianVertex(floorVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(floorVertex.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
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
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void discreteLatentsDontHaveDiffablePath() {
        GaussianVertex mu = new GaussianVertex(5., 1.);
        PoissonVertex poisson = new PoissonVertex(mu);
        boolean pathPresent = DifferentiableChecker.isDifferentiable(poisson);
        assertEquals(false, pathPresent);
    }

    @Test
    public void latentAsParentOfNonDiff() {
        GaussianVertex latent = new GaussianVertex(5., 1.);
        DoubleVertex nonDiffableVertex = new FloorVertex(latent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void observedLatentAsParentOfNonDiff() {
        GaussianVertex observedLatent = new GaussianVertex(5., 1.);
        observedLatent.observe(4);
        DoubleVertex nonDiffableVertex = new FloorVertex(observedLatent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void constantAsParentOfNonDiff() {
        DoubleVertex constantVertex = new ConstantDoubleVertex(20.);
        DoubleVertex nonDiffableVertex = new FloorVertex(constantVertex);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussian.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void NonDiffableNotOnPathToLatent() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        DoubleVertex nonDiffableVertex = new FloorVertex(gaussianA);
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussianB.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void graphWithAssertIsDiffable() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        gaussianA.lessThan(new ConstantDoubleVertex(90)).assertTrue();
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussianB.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void multipleNonDiffableWithConstantParents() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        gaussianA.observe(4.);
        gaussianB.observe(4.);

        DoubleVertex resultVertex = gaussianA.multiply(gaussianB).plus(gaussianB);
        DoubleVertex nonDiffableA = new FloorVertex(resultVertex);
        DoubleVertex nonDiffableB = new FloorVertex(resultVertex);
        DoubleVertex nonDiffableC = new FloorVertex(resultVertex);
        DoubleVertex nonDiffableD = new FloorVertex(resultVertex);
        DoubleVertex nonDiffableE = new FloorVertex(resultVertex);
        DoubleVertex nonDiffSum = nonDiffableA.plus(nonDiffableB).plus(nonDiffableC).plus(nonDiffableD).plus(nonDiffableE);
        GaussianVertex gaussianSum = new GaussianVertex(nonDiffSum, 1.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(gaussianSum.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void observedVertexWithNonDiffableParentIsntDiffable() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        FloorVertex nonDiffable = new FloorVertex(gaussianA);
        GaussianVertex observed = new GaussianVertex(nonDiffable, 1.);
        observed.observe(4.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(observed.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void observedVertexWithNonDiffableConstantParentIsDiffable() {
        DoubleVertex constDouble = new ConstantDoubleVertex(4);
        FloorVertex nonDiffable = new FloorVertex(constDouble);
        GaussianVertex observed = new GaussianVertex(nonDiffable, 1.);
        observed.observe(4.);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(observed.getConnectedGraph());
        boolean pathPresent = DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

}

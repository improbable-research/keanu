package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Test;
import org.mockito.Mockito;

import java.util.Collection;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;


public class DifferentiableCheckerTest {

    private void assertMAPIsDifferentiable(Collection<Vertex> vertices) {
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertices);
        boolean differentiable = DifferentiableChecker.isDifferentiableWrtLatents(bayesianNetwork.getLatentOrObservedVertices());
        assertTrue(differentiable);
    }

    private void assertMAPNotDifferentiable(Collection<Vertex> vertices) {
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertices);
        boolean differentiable = DifferentiableChecker.isDifferentiableWrtLatents(bayesianNetwork.getLatentOrObservedVertices());
        assertFalse(differentiable);
    }

    private void assertMLEIsDifferentiable(Collection<Vertex> vertices) {
        BayesianNetwork bayesianNetwork = new BayesianNetwork(vertices);
        boolean differentiable = DifferentiableChecker.isDifferentiableWrtLatents(bayesianNetwork.getObservedVertices());
        assertTrue(differentiable);
    }

    @Test
    public void gaussiansWithAFloorInMiddleIsntDiffable() {
        GaussianVertex a = new GaussianVertex(5., 4.);
        GaussianVertex b = new GaussianVertex(a, 1.);
        GaussianVertex c = new GaussianVertex(a, 1.);
        assertMAPIsDifferentiable(c.getConnectedGraph());
    }

    @Test
    public void simpleNonDiffable() {
        GaussianVertex latentBeforeNonDiffable = new GaussianVertex(5., 3.);
        FloorVertex nonDiffable = new FloorVertex(latentBeforeNonDiffable);
        GaussianVertex latentAfterNonDiffable = new GaussianVertex(nonDiffable, 1.);
        assertMAPNotDifferentiable(nonDiffable.getConnectedGraph());
    }

    @Test
    public void ifWithNonConstantPredicateIsntDiffable() {
        BernoulliVertex predicate = new BernoulliVertex(0.5);
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        GaussianVertex gaussianB = new GaussianVertex(5., 1.);
        DoubleVertex ifResult = If.isTrue(predicate).then(gaussianA).orElse(gaussianB);
        GaussianVertex postNonDiffVertex = new GaussianVertex(ifResult, 5);
        assertMAPNotDifferentiable(postNonDiffVertex.getConnectedGraph());
    }

    @Test
    public void ifWithConstantPredicateIsDiffable() {
        BernoulliVertex predicate = new BernoulliVertex(0.5);
        predicate.observe(true);
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        GaussianVertex gaussianB = new GaussianVertex(5., 1.);
        DoubleVertex ifResult = If.isTrue(predicate).then(gaussianA).orElse(gaussianB);
        GaussianVertex postNonDiffVertex = new GaussianVertex(ifResult, 5);
        assertMAPIsDifferentiable(postNonDiffVertex.getConnectedGraph());
    }

    @Test
    public void discreteLatentsArentDiffable() {
        GaussianVertex mu = new GaussianVertex(5., 1.);
        PoissonVertex poisson = new PoissonVertex(mu);
        assertMAPNotDifferentiable(poisson.getConnectedGraph());
    }

    @Test
    public void latentAsParentOfNonDiffIsntDiffable() {
        GaussianVertex latent = new GaussianVertex(5., 1.);
        DoubleVertex nonDiffableVertex = new FloorVertex(latent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        assertMAPNotDifferentiable(gaussian.getConnectedGraph());
    }

    @Test
    public void observedLatentAsParentOfNonDiffIsDiffable() {
        GaussianVertex observedLatent = new GaussianVertex(5., 1.);
        observedLatent.observe(4);
        FloorVertex nonDiffableVertex = new FloorVertex(observedLatent);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        assertMAPIsDifferentiable(gaussian.getConnectedGraph());
    }

    @Test
    public void constantAsParentOfNonDiffIsDiffable() {
        DoubleVertex constantVertex = new ConstantDoubleVertex(20.);
        FloorVertex nonDiffableVertex = new FloorVertex(constantVertex);
        GaussianVertex gaussian = new GaussianVertex(nonDiffableVertex, 1.);
        assertMAPIsDifferentiable(gaussian.getConnectedGraph());
    }

    @Test
    public void nonDiffableNotOnPathToLatentIsDiffable() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        FloorVertex nonDiffableVertex = new FloorVertex(gaussianA);
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        assertMAPIsDifferentiable(gaussianB.getConnectedGraph());
    }

    @Test
    public void graphWithAssertIsDiffable() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        gaussianA.lessThan(new ConstantDoubleVertex(90)).assertTrue();
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        assertMAPIsDifferentiable(gaussianB.getConnectedGraph());
    }

    @Test
    public void multipleNonDiffableWithConstantParentsIsDiffable() {
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
        assertMAPIsDifferentiable(gaussianSum.getConnectedGraph());
    }

    @Test
    public void observedVertexWithNonDiffableParentIsntDiffable() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        FloorVertex nonDiffable = new FloorVertex(gaussianA);
        GaussianVertex observed = new GaussianVertex(nonDiffable, 1.);
        observed.observe(4.);
        assertMAPNotDifferentiable(observed.getConnectedGraph());
    }

    @Test
    public void observedVertexWithNonDiffableConstantParentIsDiffable() {
        DoubleVertex constDouble = new ConstantDoubleVertex(4.);
        FloorVertex nonDiffable = new FloorVertex(constDouble);
        GaussianVertex observed = new GaussianVertex(nonDiffable, 1.);
        observed.observe(4.);
        assertMAPIsDifferentiable(observed.getConnectedGraph());
    }

    @Test
    public void graphWhichShouldBeMLEDiffableAndNotMAPDiffable() {
        DoubleVertex constDouble = new ConstantDoubleVertex(5.);
        GaussianVertex gaussianA = new GaussianVertex(constDouble, constDouble);
        FloorVertex nonDiffable = new FloorVertex(gaussianA);
        GaussianVertex gaussianB = new GaussianVertex(nonDiffable, constDouble);
        GaussianVertex gaussianObserved = new GaussianVertex(gaussianB, constDouble);

        // Graph has no non diffable vertices between observed and next latent. So is MLE diffable.
        assertMLEIsDifferentiable(gaussianObserved.getConnectedGraph());

        // A non diffable floor is between latent and the next latent. So not MAP diffable.
        assertMAPNotDifferentiable(gaussianObserved.getConnectedGraph());
    }

    @Test
    public void constantVerticesAreCached() {
        GaussianVertex baseVertex = new GaussianVertex(1., 1.);
        AdditionVertex addVertex = new AdditionVertex(baseVertex, new ConstantDoubleVertex(1.));
        DoubleVertex mockedVertex = Mockito.spy(addVertex);
        FloorVertex nonDiffable = new FloorVertex(mockedVertex);

        GaussianVertex gaussianA = new GaussianVertex(nonDiffable, new ConstantDoubleVertex(3.));
        GaussianVertex gaussianB = new GaussianVertex(nonDiffable, new ConstantDoubleVertex(3.));
        DifferentiableChecker.isDifferentiableWrtLatents(gaussianB.getConnectedGraph());

        /*
        After gaussianA is checked for whether it is diffable we know that FloorVertex is constant.
        This should be cached and therefore its parents aren't needed to be explored to determine this in the future.
        isObserved is called when checking for a constant value, so it should only be called once.
        */
        verify(mockedVertex, times(1)).isObserved();
    }
}

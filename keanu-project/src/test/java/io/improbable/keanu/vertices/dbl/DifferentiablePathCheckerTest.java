package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Test;

import static org.junit.Assert.assertEquals;


public class DifferentiablePathCheckerTest {

    @Test
    public void testBasicDiffPath() {
        GaussianVertex a = new GaussianVertex(5, 4);
        GaussianVertex b = new GaussianVertex(a, 1);
        GaussianVertex c = new GaussianVertex(a, 1);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(c.getConnectedGraph());
        DifferentiablePathChecker dpc = new DifferentiablePathChecker(bayesianNetwork.getContinuousLatentVertices());
        boolean pathPresent = dpc.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(true, pathPresent);
    }

    @Test
    public void testBasicDiffFail() {
        GaussianVertex latentBeforeNonDiffOp = new GaussianVertex(5, 3);
        DoubleVertex floorVertex = new FloorVertex(latentBeforeNonDiffOp);
        GaussianVertex latentAfterNonDiffOp = new GaussianVertex(floorVertex, 1);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(floorVertex.getConnectedGraph());
        DifferentiablePathChecker dpc = new DifferentiablePathChecker(bayesianNetwork.getContinuousLatentVertices());
        boolean pathPresent = dpc.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }

    @Test
    public void IfWithNonConstantPredicateIsntDiffable() {
        BernoulliVertex predicate = new BernoulliVertex(0.5);
        GaussianVertex gaussianA = new GaussianVertex(5, 1);
        GaussianVertex gaussianB = new GaussianVertex(5, 1);
        DoubleVertex ifResult = If.isTrue(predicate).then(gaussianA).orElse(gaussianB);
        GaussianVertex postNonDiffVertex = new GaussianVertex(ifResult, 5);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(postNonDiffVertex.getConnectedGraph());
        DifferentiablePathChecker dpc = new DifferentiablePathChecker(bayesianNetwork.getContinuousLatentVertices());
        boolean pathPresent = dpc.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false, pathPresent);
    }
}

package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
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
        GaussianVertex gaussianVertex = new GaussianVertex(5,3);
        DoubleVertex floorVertex = new FloorVertex(gaussianVertex);
        GaussianVertex gaussianVertex2 = new GaussianVertex(floorVertex,1);
        BayesianNetwork bayesianNetwork = new BayesianNetwork(floorVertex.getConnectedGraph());
        DifferentiablePathChecker dpc = new DifferentiablePathChecker(bayesianNetwork.getContinuousLatentVertices());
        boolean pathPresent = dpc.differentiablePath(bayesianNetwork.getLatentOrObservedVertices());
        assertEquals(false,pathPresent);
    }
}

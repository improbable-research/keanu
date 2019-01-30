package io.improbable.keanu.util.graph;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BuildGraphTest {

    private static DoubleVertex complexResultVertex;

    @BeforeClass
    public static void setUpComplexNet() {
        VertexId.ID_GENERATOR.set(0);
        complexResultVertex =
            new GaussianVertex(
                new GammaVertex(0, 1)
                    .plus(new GaussianVertex(0, 1))
                    .plus(new ConstantDoubleVertex(5)),
                new ConstantDoubleVertex(0.2));
    }

    @Before
    public void resetVertexIdsAndOutputStream() {
        VertexId.ID_GENERATOR.set(0);
    }

    @Test
    public void buildFromBayesianNetwork() {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        VertexGraph graph = new VertexGraph(complexNet);
        graph.labelEdgesWithParameters();
        assertEquals(10, graph.edgeCount());
        assertEquals(11, graph.nodeCount());
    }

    @Test
    public void buildFromVertex() {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters();
        assertEquals(10, graph.edgeCount());
        assertEquals(11, graph.nodeCount());
    }

    @Test
    public void buildFromVertexDegree() {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        VertexGraph graph = new VertexGraph(complexNet, complexResultVertex, 4);
        graph.labelEdgesWithParameters();
        assertEquals(10, graph.edgeCount());
        assertEquals(11, graph.nodeCount());
    }

    @Test
    public void transformNetworkRemovingDetermanisticNodes() {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters();
        graph.removeDeterministicVertices();
        assertEquals(2, graph.edgeCount());
        assertEquals(3, graph.nodeCount());
    }
}

package io.improbable.keanu.network;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ParetoVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class ModelCompositionTest {

    private BayesianNetwork innerNet;
    private UniformVertex trueLocation;
    private BayesianNetwork outerNet;
    private DoubleVertex location;
    private DoubleVertex size;
    private DoubleVertex gaussian;
    private DoubleVertex pareto;
    private DoubleVertex gaussOutputVertex;
    private DoubleVertex paretoOutputVertex;

    @Before
    public void setup() {

        createInnerNet();

        trueLocation = new UniformVertex(-50.0, 50.0);

        Map<VertexLabel, Vertex> inputVertices = ImmutableMap.of(new VertexLabel("Location"), trueLocation);

        Map<VertexLabel, Vertex> outputs = ModelComposition.createModelVertices(
            innerNet, inputVertices, ImmutableList.of(new VertexLabel("Output1"), new VertexLabel("Output2")));
        gaussOutputVertex = (DoubleVertex)outputs.get(new VertexLabel("Output1"));
        paretoOutputVertex = (DoubleVertex)outputs.get(new VertexLabel("Output2"));

        outerNet = new BayesianNetwork(paretoOutputVertex.getConnectedGraph());
    }

    private void createInnerNet() {
        location = new DoubleProxyVertex();
        location.setLabel(new VertexLabel("Location"));
        size = new UniformVertex(0.1, 20);
        size.setLabel(new VertexLabel("Size"));
        gaussian = new GaussianVertex(location, size);
        gaussian.setLabel(new VertexLabel("Output1"));
        pareto = new ParetoVertex(location, size);
        pareto.setLabel(new VertexLabel("Output2"));

        innerNet = new BayesianNetwork(gaussian.getConnectedGraph());
    }

    @Test
    public void outputVerticesCorrectlyReturned() {
        assertEquals(gaussOutputVertex, gaussian);
        assertEquals(paretoOutputVertex, pareto);
    }

    @Test
    public void verticesAreAtCorrectDepth() {
        assertEquals(trueLocation.getId().getDepth(), 1);
        assertEquals(location.getId().getDepth(), 2);
        assertEquals(size.getId().getDepth(), 2);
        assertEquals(gaussian.getId().getDepth(), 1);
        assertEquals(pareto.getId().getDepth(), 1);
    }

    @Test
    public void netsAtCorrectDepth() {
        assertEquals(innerNet.getDepth(), 2);
        assertEquals(outerNet.getDepth(), 1);
    }

    @Test
    public void idOrderingStillImpliesTopologicalOrdering() {
        for (Vertex v: outerNet.getAllVertices()) {
            Set<Vertex> parentSet = v.getParents();
            for (Vertex parent : parentSet) {
                assertTrue(v.getId().compareTo(parent.getId()) > 0);
            }
        }
    }

    @Test
    public void inferenceWorksOnGraph() {
        final int NUM_SAMPLES = 10000;
        final double REAL_HYPER_LOC = 5.1;
        final double REAL_HYPER_SIZE = 1.0;

        trueLocation.setValue(9.9);
        size.setValue(9.9);
        GaussianVertex sourceOfTruth = new GaussianVertex(new int[]{NUM_SAMPLES, 1}, REAL_HYPER_LOC, REAL_HYPER_SIZE);

        gaussOutputVertex.observe(sourceOfTruth.sample());
        GradientOptimizer optimizer = GradientOptimizer.of(outerNet);
        optimizer.maxAPosteriori();

        assertEquals(trueLocation.getValue().scalar(), REAL_HYPER_LOC, 0.05);
        assertEquals(size.getValue().scalar(), REAL_HYPER_SIZE, 0.05);
    }

    @Test
    public void bayesNetCanFilterOnDepth() {
        Set<Vertex> latentOuterVertices = ImmutableSet.of(trueLocation, paretoOutputVertex, gaussOutputVertex);
        Set<Vertex> filteredVertices = new HashSet<>(outerNet.getLatentVerticesAtDepth(outerNet.getDepth()));

        assertEquals(latentOuterVertices.containsAll(filteredVertices), true);
    }

}

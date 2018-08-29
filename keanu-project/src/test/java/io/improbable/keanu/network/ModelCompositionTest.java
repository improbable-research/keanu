package io.improbable.keanu.network;

import static org.junit.Assert.assertEquals;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

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
    private UniformVertex trueLoc;
    private UniformVertex trueSize;
    private BayesianNetwork outerNet;
    private DoubleVertex loc;
    private DoubleVertex size;
    private DoubleVertex gaussian;
    private DoubleVertex pareto;
    private DoubleVertex gaussOutputVertex;
    private DoubleVertex paretoOutputVertex;

    @Before
    public void setup() {
        loc = new DoubleProxyVertex();
        loc.setLabel(new VertexLabel("Loc"));
        size = new DoubleProxyVertex();
        size.setLabel(new VertexLabel("Size"));
        gaussian = new GaussianVertex(loc, size);
        gaussian.setLabel(new VertexLabel("Output1"));
        pareto = new ParetoVertex(loc, size);
        pareto.setLabel(new VertexLabel("Output2"));

        innerNet = new BayesianNetwork(gaussian.getConnectedGraph());

        trueLoc = new UniformVertex(5.0, 10.0);
        trueSize = new UniformVertex(0.1, 10.0);

        Map<VertexLabel, Vertex> inputVertices = ImmutableMap.of(
            new VertexLabel("Loc"), trueLoc,
            new VertexLabel("Size"), trueSize);

        Map<VertexLabel, Vertex> outputs = ModelComposition.createModelVertices(
            innerNet, inputVertices, ImmutableList.of(new VertexLabel("Output1"), new VertexLabel("Output2")));
        gaussOutputVertex = (DoubleVertex)outputs.get(new VertexLabel("Output1"));
        paretoOutputVertex = (DoubleVertex)outputs.get(new VertexLabel("Output2"));

        outerNet = new BayesianNetwork(paretoOutputVertex.getConnectedGraph());
    }

    @Test
    public void outputVerticesCorrectlyReturned() {
        assertEquals(gaussOutputVertex, gaussian);
        assertEquals(paretoOutputVertex, pareto);
    }

    @Test
    public void verticesAreAtCorrectDepth() {
        assertEquals(trueLoc.getId().getDepth(), 1);
        assertEquals(trueSize.getId().getDepth(), 1);
        assertEquals(loc.getId().getDepth(), 2);
        assertEquals(size.getId().getDepth(), 2);
        assertEquals(gaussian.getId().getDepth(), 2);
        assertEquals(pareto.getId().getDepth(), 2);
    }

    @Test
    public void idOrderingStillImpliesTopologicalOrdering() {

    }

    @Test
    public void inferenceWorksOnGraph() {
        final int NUM_SAMPLES = 10000;
        final double REAL_HYPER_LOC = 5.1;
        final double REAL_HYPER_SIZE = 1.0;

        trueLoc.setValue(9.9);
        trueSize.setValue(9.9);
        GaussianVertex sourceOfTruth = new GaussianVertex(new int[]{NUM_SAMPLES, 1}, REAL_HYPER_LOC, REAL_HYPER_SIZE);

        gaussOutputVertex.observe(sourceOfTruth.sample());
        GradientOptimizer optimizer = GradientOptimizer.of(outerNet);
        optimizer.maxAPosteriori();

        assertEquals(trueLoc.getValue().scalar(), REAL_HYPER_LOC, 0.05);
        assertEquals(trueSize.getValue().scalar(), REAL_HYPER_SIZE, 0.05);
    }

}

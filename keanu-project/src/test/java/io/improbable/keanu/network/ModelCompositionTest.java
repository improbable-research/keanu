package io.improbable.keanu.network;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
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

    /*
     * These tests run over a single graph within a graph.  The "inner" graph has a single Proxy input for the location
     * a probabilistic value for size and two potential outputs:
     *
     * | Proxy "Location" |              | Uniform - Size |
     *                     \            /
     *     | Gaussian ("Output1") & Pareto ("Output2") |
     *
     * The Outer graph hooks up a Uniform to the "Location" input of the inner graph and takes the Output "Output1" as
     * the single output it cares about.
     */
    @Before
    public void setup() {

        innerNet = createInnerNet();

        trueLocation = new UniformVertex(0.1, 50.0);

        Map<VertexLabel, Vertex> inputVertices = ImmutableMap.of(new VertexLabel("Location"), trueLocation);

        Map<VertexLabel, Vertex> outputs = ModelComposition.composeModel(
            innerNet, inputVertices, ImmutableList.of(new VertexLabel("Output1")));
        gaussOutputVertex = (DoubleVertex)outputs.get(new VertexLabel("Output1"));

        outerNet = new BayesianNetwork(gaussOutputVertex.getConnectedGraph());
    }

    private BayesianNetwork createInnerNet() {
        location = new DoubleProxyVertex();
        location.setLabel(new VertexLabel("Location"));
        size = new UniformVertex(0.1, 20);
        size.setLabel(new VertexLabel("Size"));
        gaussian = new GaussianVertex(location, size);
        gaussian.setLabel(new VertexLabel("Output1"));
        pareto = new ParetoVertex(location, size);
        pareto.setLabel(new VertexLabel("Output2"));

        return new BayesianNetwork(gaussian.getConnectedGraph());
    }

    @Test
    public void outputVerticesCorrectlyReturned() {
        assertEquals(gaussOutputVertex, gaussian);
    }

    @Test
    public void verticesAreAtCorrectDepth() {
        assertEquals(trueLocation.getId().getIndentation(), 1);
        assertEquals(location.getId().getIndentation(), 2);
        assertEquals(size.getId().getIndentation(), 2);
        assertEquals(gaussian.getId().getIndentation(), 1);
        assertEquals(pareto.getId().getIndentation(), 2);
    }

    @Test
    public void netsAtCorrectDepth() {
        assertEquals(innerNet.getIndentation(), 2);
        assertEquals(outerNet.getIndentation(), 1);
    }

    @Test
    public void idOrderingStillImpliesTopologicalOrdering() {
        for (Vertex v: outerNet.getVertices()) {
            Set<Vertex> parentSet = v.getParents();
            for (Vertex parent : parentSet) {
                assertTrue(v.getId().compareTo(parent.getId()) > 0);
            }
        }
    }

    @Test
    public void inferenceWorksOnGraph() {
        final int NUM_SAMPLES = 50000;
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
    public void bayesNetCanReturnTopLevelVerticesOnly() {
        Set<Vertex> latentOuterVertices = ImmutableSet.of(trueLocation, gaussOutputVertex);
        Set<Vertex> filteredVertices = new HashSet<>(outerNet.getTopLevelLatentVertices());

        assertEquals(latentOuterVertices.containsAll(filteredVertices), true);
        assertEquals(filteredVertices.containsAll(latentOuterVertices), true);
    }

    private void triggerCompositionWarnings(List<VertexLabel> outputs,
                                            String inputLabel,
                                            String outputLabel,
                                            String plusLabel) {
        DoubleVertex inputVertex = new DoubleProxyVertex();
        inputVertex.setLabel(new VertexLabel(inputLabel));
        DoubleVertex plusValue = new ConstantDoubleVertex(1.0);
        plusValue.setLabel(new VertexLabel(plusLabel));
        DoubleVertex outputVertex = inputVertex.plus(plusValue);
        outputVertex.setLabel(new VertexLabel(outputLabel));
        BayesianNetwork bayesNet = new BayesianNetwork(outputVertex.getConnectedGraph());
        DoubleVertex outerInput = new GaussianVertex(0.0, 1.0);

        ModelComposition.composeModel(bayesNet,
            ImmutableMap.of(new VertexLabel("Input1"), outerInput),
            outputs);
    }

    @Test(expected = IllegalArgumentException.class)
    public void willRejectEmptyOutput() {
        triggerCompositionWarnings(new ArrayList<>(), "Input1", "Output1", "Plus");
    }

    @Test(expected = IllegalArgumentException.class)
    public void willRejectMissingOutput() {
        triggerCompositionWarnings(ImmutableList.of(new VertexLabel("Invalid Label")),
            "Input1", "Output1", "Plus");
    }

    @Test(expected = IllegalArgumentException.class)
    public void willRejectMissingInput() {
        triggerCompositionWarnings(ImmutableList.of(new VertexLabel("Output1")),
            "Random Name", "Output1", "Plus");
    }

    @Test(expected = IllegalArgumentException.class)
    public void willRejectNonProxyInput() {
        triggerCompositionWarnings(ImmutableList.of(new VertexLabel("Output1")),
            "Random Name", "Output1", "Input1");
    }

    @Test(expected = IllegalArgumentException.class)
    public void willRejectProxyWithParent() {
        DoubleProxyVertex proxy = new DoubleProxyVertex();
        proxy.setLabel(new VertexLabel("Input1"));
        DoubleVertex output = new GaussianVertex(proxy, 1.0);
        output.setLabel(new VertexLabel("Output1"));
        DoubleVertex outerInput = new UniformVertex(-1.0, 1.0);
        DoubleVertex invalidParent = new ConstantDoubleVertex(0.0);
        proxy.setParent(invalidParent);

        BayesianNetwork bayesNet = new BayesianNetwork(output.getConnectedGraph());
        ModelComposition.composeModel(bayesNet,
            ImmutableMap.of(new VertexLabel("Input1"), outerInput),
            ImmutableList.of(new VertexLabel("Output1")));
    }

    @Test
    public void vertexIdPrefixMatchingWorks() {
        VertexId prefix = new VertexId();
        VertexId shouldMatch = new VertexId();
        shouldMatch.addPrefix(prefix);
        VertexId shouldNotMatch = new VertexId();

        assertEquals(shouldMatch.prefixMatches(prefix), true);
        assertEquals(shouldNotMatch.prefixMatches(prefix), false);
    }

}

package io.improbable.keanu.network;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.Mockito.mock;

public class BayesianNetworkTest {

    BayesianNetwork network;
    Set<Vertex> connectedGraph;
    BooleanVertex input1;
    BooleanVertex input2;
    BooleanVertex output;
    final String LABEL_A = "Label A";
    final String LABEL_B = "Label B";
    final String LABEL_ORED = "Output";

    @Before
    public void setUpNetwork() {
        input1 = new BernoulliVertex(0.25);
        input2 = new BernoulliVertex(0.75);
        output = input1.or(input2);
        connectedGraph = output.getConnectedGraph();
        network = new BayesianNetwork(connectedGraph);
    }

    @Test
    public void youCanObserveAProbabilisticVariableAfterCreatingTheNetwork() {
        assertThat(network.getObservedVertices(), is(empty()));
        input1.observe(true);
        assertThat(network.getObservedVertices(), contains(input1));
        input2.observe(true);
        assertThat(network.getObservedVertices(), containsInAnyOrder(input1, input2));
        input1.unobserve();
        assertThat(network.getObservedVertices(), contains(input2));
    }

    @Test
    public void youCanObserveANonProbabilisticVariableAfterCreatingTheNetwork() {
        assertThat(network.getObservedVertices(), is(empty()));
        output.observe(true);
        assertThat(network.getObservedVertices(), contains(output));
        output.unobserve();
        assertThat(network.getObservedVertices(), is(empty()));
    }

    @Test
    public void youCanLabelVertices() {
        BooleanVertex a = new BernoulliVertex(0.5);
        BooleanVertex b = new BernoulliVertex(0.5);
        BooleanVertex ored = a.or(b);
        BooleanVertex unlabelled = ored.or(a);
        Vertex retrieved;
        VertexLabel labelA = new VertexLabel(LABEL_A);
        VertexLabel labelB = new VertexLabel(LABEL_B);
        VertexLabel labelOr = new VertexLabel(LABEL_ORED);

        a.setLabel(labelA);
        b.setLabel(labelB);
        ored.setLabel(labelOr);

        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());
        retrieved = net.getVertexByLabel(labelA);
        assertThat(retrieved, is(a));
        retrieved = net.getVertexByLabel(labelB);
        assertThat(retrieved, is(b));
        retrieved = net.getVertexByLabel(labelOr);
        assertThat(retrieved, is(ored));
        retrieved = net.getVertexByLabel(null);
        assertThat(retrieved, nullValue());
    }

    @Test(expected = IllegalArgumentException.class)
    public void labelErrorsDetected() {
        BooleanVertex a = new BernoulliVertex(0.5);
        BooleanVertex b = new BernoulliVertex(0.5);
        BooleanVertex ored = a.or(b);

        a.setLabel(LABEL_A);
        b.setLabel(LABEL_A);

        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());
    }

    private class TestNonSaveableVertex extends DoubleVertex implements NonSaveableVertex {
        @Override
        public DoubleTensor sample(KeanuRandom random) {
            return null;
        }

        @Override
        public DoubleTensor sample() {
            return null;
        }

        private TestNonSaveableVertex() {
            super(new long[]{1, 1});
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void networkWithNonSaveableVerticesThrowsExceptionOnSave() throws IOException {
        DoubleVertex testVertex = new TestNonSaveableVertex();
        BayesianNetwork net = new BayesianNetwork(testVertex.getConnectedGraph());
        NetworkSaver netSaver = mock(NetworkSaver.class);
        net.save(netSaver);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cantInstantiateEmptyBayesianNetwork() {
        BayesianNetwork net = new BayesianNetwork(new HashSet<>());
    }

    @Test
    public void testGetNumVertices() {
        assertThat(network.getVertexCount(), equalTo(connectedGraph.size()));
    }

    @Test
    public void testGetAverageVertexDegree() {
        assertThat(network.getAverageVertexDegree(), equalTo((1. + 1. + 2. + 2. + 2.) / 5));
    }

    @Test
    public void networkReturnsVerticesInNamespace() {
        BooleanVertex a0 = new BernoulliVertex(0.5);
        BooleanVertex a1 = new BernoulliVertex(0.5);
        BooleanVertex b0 = new BernoulliVertex(0.5);
        BooleanVertex c = new BernoulliVertex(0.5);

        a0.setLabel(new VertexLabel("root", "a", "0"));
        a1.setLabel(new VertexLabel("root", "a", "1"));
        b0.setLabel(new VertexLabel("root", "b", "0"));

        BayesianNetwork net = new BayesianNetwork(Arrays.asList(a0, a1, b0, c));
        List<Vertex> verticesInNamespace = net.getVerticesInNamespace("root");

        assertThat(verticesInNamespace.size(), equalTo(3));
        assertTrue(verticesInNamespace.containsAll(Arrays.asList(a0, a1, b0)));
    }

}
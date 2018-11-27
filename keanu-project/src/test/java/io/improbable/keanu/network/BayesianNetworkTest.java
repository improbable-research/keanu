package io.improbable.keanu.network;

import io.improbable.keanu.plating.loop.Loop;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BoolProxyVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import static org.hamcrest.Matchers.equalTo;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Set;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.Mockito.mock;

public class BayesianNetworkTest {

    BayesianNetwork network;
    BoolVertex input1;
    BoolVertex input2;
    BoolVertex output;
    final String LABEL_A = "Label A";
    final String LABEL_B = "Label B";
    final String LABEL_ORED = "Output";

    @Before
    public void setUpNetwork() {
        input1 = new BernoulliVertex(0.25);
        input2 = new BernoulliVertex(0.75);
        output = input1.or(input2);
        network = new BayesianNetwork(output.getConnectedGraph());
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
        BoolVertex a = new BernoulliVertex(0.5);
        BoolVertex b = new BernoulliVertex(0.5);
        BoolVertex ored = a.or(b);
        BoolVertex unlabelled = ored.or(a);
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
        BoolVertex a = new BernoulliVertex(0.5);
        BoolVertex b = new BernoulliVertex(0.5);
        BoolVertex ored = a.or(b);

        a.setLabel(LABEL_A);
        b.setLabel(LABEL_A);

        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());
    }

    @Test(expected = IllegalArgumentException.class)
    public void networkWithNonSaveableVerticesThrowsExceptionOnSave() throws IOException {
        BoolVertex testVertex = new BoolProxyVertex(new VertexLabel("test_vertex"));
        BayesianNetwork net = new BayesianNetwork(testVertex.getConnectedGraph());
        NetworkSaver netSaver = mock(NetworkSaver.class);
        net.save(netSaver);
    }

    @Test
    public void youCanGetVertexCountOfInitialBayesNet() {
        ConstantDoubleVertex probTrue = ConstantVertex.of(0.5);
        BoolVertex a = new BernoulliVertex(probTrue);
        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());

        BoolVertex b = new BernoulliVertex(0.5);
        BoolVertex ored = a.or(b);

        assertThat(net.getVertexCount(), equalTo(2));
    }

    @Test
    public void youCanGetVertexDegreeOfInitialBayesNet() {
        ConstantDoubleVertex probTrue = ConstantVertex.of(0.5);
        BoolVertex a = new BernoulliVertex(probTrue);
        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());

        BoolVertex b = new BernoulliVertex(0.5);
        BoolVertex ored = a.or(b);

        assertThat(net.getVertexDegree(a.getId()), equalTo(1));
        assertThat(net.getVertexDegree(a), equalTo(1));
    }

    @Test
    public void youCanCalculateAverageVertexDegreeOfInitialBayesNet() {
        ConstantDoubleVertex mu = ConstantVertex.of(0.);
        ConstantDoubleVertex sigma = ConstantVertex.of(1.);
        GaussianVertex a = new GaussianVertex(mu, sigma);

        GaussianVertex b = new GaussianVertex(a, sigma);
        GaussianVertex c = new GaussianVertex(a, sigma);

        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());

        GaussianVertex d = new GaussianVertex(a, 1.);
        GaussianVertex e = new GaussianVertex(a, 1.);

        assertThat(net.getAverageVertexDegree(), equalTo((1. + 3. + 4. + 2. + 2.) / 5.));
    }

    @Test
    public void youCanCalculateMetricsOfVerticesInALoop() {
         Loop loop = Loop
            .withInitialConditions(ConstantVertex.of(0.))
            .iterateWhile(() -> new BernoulliVertex(ConstantVertex.of(0.5)))
            .apply((v) -> v.plus(ConstantVertex.of(1.)));
         Vertex<?> output = loop.getOutput();

         Set<Vertex> connectedGraphInLoop = output.getConnectedGraph();
         BayesianNetwork net = new BayesianNetwork(connectedGraphInLoop);

         assertThat(net.getVertexCount(), equalTo(connectedGraphInLoop.size()));
         assertThat(net.getVertexDegree(output), equalTo(output.getDegree()));
         assertThat(net.getAverageVertexDegree(), equalTo(connectedGraphInLoop.stream().mapToInt(Vertex::getDegree).average().getAsDouble()));
    }
}

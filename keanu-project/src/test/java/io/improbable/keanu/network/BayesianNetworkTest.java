package io.improbable.keanu.network;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;

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
        BoolVertex or2 = ored.or(a);
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

        a.setLabel(new VertexLabel(LABEL_A));
        b.setLabel(new VertexLabel(LABEL_A));

        BayesianNetwork net = new BayesianNetwork(a.getConnectedGraph());
    }
}

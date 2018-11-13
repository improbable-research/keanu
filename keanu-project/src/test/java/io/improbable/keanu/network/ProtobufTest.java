package io.improbable.keanu.network;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;

public class ProtobufTest {
    
    @Test
    public void youCanSaveAndLoadANetworkWithValues() throws IOException {
        DoubleVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        BayesianNetwork net = new BayesianNetwork(gaussianVertex.getConnectedGraph());
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        ProtobufWriter protobufWriter = new ProtobufWriter(net);
        protobufWriter.save(output, true);
        assertThat(output.size(), greaterThan(0));
        ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());

        ProtobufReader reader = new ProtobufReader();
        BayesianNetwork readNet = reader.loadNetwork(input);

        assertThat(readNet.getLatentVertices().size(), is(1));
        assertThat(readNet.getLatentVertices().get(0), instanceOf(GaussianVertex.class));
        GaussianVertex readGaussianVertex = (GaussianVertex)readNet.getLatentVertices().get(0);
        assertThat(readGaussianVertex.getMu().getValue().scalar(), closeTo(0.0, 1e-10));
        assertThat(readGaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        readGaussianVertex.sample();

    }
}

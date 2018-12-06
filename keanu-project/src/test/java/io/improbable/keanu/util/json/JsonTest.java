package io.improbable.keanu.util.json;

import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.collection.IsIn.isIn;
import static org.hamcrest.core.Every.everyItem;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

public class JsonTest {

    private static BayesianNetwork net;
    private static Map<String, String> someMetadata;
    private static ByteArrayOutputStream output;

    @BeforeClass
    public static void setup() throws IOException {
        DoubleVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        gaussianVertex.observe(0.5);
        net = new BayesianNetwork(gaussianVertex.getConnectedGraph());

        output = new ByteArrayOutputStream();

        someMetadata = new HashMap<>();
        someMetadata.put("Author", "Some Author");
        someMetadata.put("Tag", "MyBayesNet");

        JsonSaver jasonSaver = new JsonSaver(net, someMetadata);
        jasonSaver.save(output, true);
    }

    @Test
    public void metadataIsSavedForJsonModels() throws IOException {
        KeanuSavedBayesNet.Metadata.Builder metadataBuilder = KeanuSavedBayesNet.Metadata.newBuilder();
        for (Map.Entry<String, String> entry : someMetadata.entrySet()) {
            metadataBuilder.putMetadataInfo(entry.getKey(), entry.getValue());
        }
        KeanuSavedBayesNet.Model.Builder modelBuilder = KeanuSavedBayesNet.Model.newBuilder();
        JsonFormat.parser().merge(output.toString(), modelBuilder);
        KeanuSavedBayesNet.Model parsedModel = modelBuilder.build();

        assertTrue(parsedModel.hasMetadata());
        assertThat(parsedModel.getMetadata().getMetadataInfoMap().entrySet(), everyItem(isIn(metadataBuilder.getMetadataInfoMap().entrySet())));
    }

    @Test
    public void modelsCanBeSavedAsJsonAndReadBackIn() throws IOException {
        ByteArrayInputStream input = new ByteArrayInputStream(output.toByteArray());
        JsonLoader jsonLoader = new JsonLoader();
        BayesianNetwork loadedNetwork = jsonLoader.loadNetwork(input);

        assertNotEquals(net, loadedNetwork);
        assertEquals(net.getAllVertices().size(), loadedNetwork.getAllVertices().size());
        GaussianVertex gaussianVertex = (GaussianVertex)loadedNetwork.getLatentVertices().get(0);
        assertThat(gaussianVertex.getMu().getValue().scalar(), closeTo(0.0, 1e-10));
        assertThat(gaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        assertThat(gaussianVertex.getValue().scalar(), closeTo(0.5, 1e-10));
    }
}

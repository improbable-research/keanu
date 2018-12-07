package io.improbable.keanu.util.json;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Resources;
import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.collection.IsIn.isIn;
import static org.hamcrest.core.Every.everyItem;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class JsonTest {

    private static final String JSON_EXPECTED_OUTPUT_FILE = "jsonFiles/JsonModelOutput.json";

    private static BayesianNetwork net;
    private static Map<String, String> someMetadata;
    private static ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    @BeforeClass
    public static void setup() throws IOException {
        DoubleVertex gaussianVertex = new GaussianVertex(0.0, 1.0);
        gaussianVertex.observe(0.5);
        gaussianVertex.setLabel("GaussianVertex");
        net = new BayesianNetwork(gaussianVertex.getConnectedGraph());

        someMetadata = ImmutableMap.of( "Author", "Some Author", "Tag", "MyBayesNet" );
        JsonSaver jsonSaver = new JsonSaver(net, someMetadata);
        jsonSaver.save(outputStream, true);
    }

    @Test
    public void jsonSaverCreatesExpectedJsonOutput() throws IOException {
        URL url = Resources.getResource(JSON_EXPECTED_OUTPUT_FILE);
        String expectedOutput = Resources.toString(url, Charsets.UTF_8);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode expectedJsonObject = mapper.readTree(expectedOutput);
        JsonNode actualJsonObject = mapper.readTree(outputStream.toString());
        assertEquals(expectedJsonObject, actualJsonObject);
    }

    @Test
    public void jsonSaverSavesMetadata() throws IOException {
        KeanuSavedBayesNet.Metadata.Builder metadataBuilder = KeanuSavedBayesNet.Metadata.newBuilder();
        for (Map.Entry<String, String> entry : someMetadata.entrySet()) {
            metadataBuilder.putMetadataInfo(entry.getKey(), entry.getValue());
        }
        KeanuSavedBayesNet.Model.Builder modelBuilder = KeanuSavedBayesNet.Model.newBuilder();
        JsonFormat.parser().merge(outputStream.toString(), modelBuilder);
        KeanuSavedBayesNet.Model parsedModel = modelBuilder.build();

        assertTrue(parsedModel.hasMetadata());
        assertEquals(parsedModel.getMetadata().getMetadataInfoMap().size(), (metadataBuilder.getMetadataInfoMap().size()));
        assertThat(parsedModel.getMetadata().getMetadataInfoMap().entrySet(), everyItem(isIn(metadataBuilder.getMetadataInfoMap().entrySet())));
    }

    @Test
    public void modelCanBeLoadedFromJson() throws IOException {
        ByteArrayInputStream input = new ByteArrayInputStream(outputStream.toByteArray());
        JsonLoader jsonLoader = new JsonLoader();
        BayesianNetwork loadedNetwork = jsonLoader.loadNetwork(input);

        assertEquals(net.getAllVertices().size(), loadedNetwork.getAllVertices().size());
        GaussianVertex gaussianVertex = (GaussianVertex)loadedNetwork.getLatentVertices().get(0);
        assertThat(gaussianVertex.getMu().getValue().scalar(), closeTo(0.0, 1e-10));
        assertThat(gaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        assertThat(gaussianVertex.getValue().scalar(), closeTo(0.5, 1e-10));
    }
}

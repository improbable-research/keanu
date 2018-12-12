package io.improbable.keanu.util.json;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Resources;
import com.google.protobuf.util.JsonFormat;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Map;

import static org.hamcrest.CoreMatchers.instanceOf;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.collection.IsIn.isIn;
import static org.hamcrest.core.Every.everyItem;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class JsonTest {

    private static final String JSON_EXPECTED_OUTPUT_FILE = "jsonFiles/JsonModelOutput.json";
    private static final String JSON_INVALID_VERTEX_TYPE_FILE = "jsonFiles/InvalidVertexType.json";
    private static final String JSON_MISSING_PARENT_VERTEX_FILE = "jsonFiles/ParentVertexMissing.json";
    private static final String JSON_MISSING_HYPERPARAMETER_FILE = "jsonFiles/HyperparameterSpecificationMissing.json";
    private static final String JSON_WRONG_ARGUMENT_TYPE_FILE = "jsonFiles/WrongArgumentType.json";
    private static final String JSON_INCOMPATIBLE_LABELS_FILE = "jsonFiles/IncompatibleLabels.json";
    private static final String JSON_INCOMPATIBLE_VALUE_FILE = "jsonFiles/IncompatibleValueSpecified.json";

    private static BayesianNetwork net;
    private static Map<String, String> someMetadata;
    private static ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    @Rule
    public ExpectedException expectedException = ExpectedException.none();

    @BeforeClass
    public static void setup() throws IOException {
        VertexId.ID_GENERATOR.set(0);

        DoubleVertex mu = new ConstantDoubleVertex(0);
        DoubleVertex sigma = new ConstantDoubleVertex(new double[]{3.0, 4.0});
        DoubleVertex gaussianVertex = new GaussianVertex(mu, sigma);
        gaussianVertex.observe(0.5);
        gaussianVertex.setLabel("GaussianVertex");
        net = new BayesianNetwork(gaussianVertex.getConnectedGraph());

        someMetadata = ImmutableMap.of( "Author", "Some Author", "Tag", "MyBayesNet" );
        JsonSaver jsonSaver = new JsonSaver(net);
        jsonSaver.save(outputStream, true, someMetadata);
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
        assertEquals(net.getLatentVertices().size(), loadedNetwork.getLatentVertices().size());
        assertEquals(net.getObservedVertices().size(), loadedNetwork.getObservedVertices().size());
        assertThat(net.getLatentVertices().get(0), instanceOf(GaussianVertex.class));
        GaussianVertex gaussianVertex = (GaussianVertex)loadedNetwork.getLatentVertices().get(0);
        assertThat(gaussianVertex.getMu().getValue().scalar(), closeTo(0.0, 1e-10));
        assertThat(gaussianVertex.getSigma().getValue().scalar(), closeTo(1.0, 1e-10));
        assertThat(gaussianVertex.getValue().scalar(), closeTo(0.5, 1e-10));
    }

    @Test
    public void loadFailsIfInvalidVertexSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Unknown Vertex Type Specified: io.improbable.keanu.vertices.dbl.nonprobabilistic.NonExistentVertexType");
        tryLoadingNetwork(JSON_INVALID_VERTEX_TYPE_FILE);
    }

    @Test
    public void loadFailsIfParentVertexInfoIsMissing() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parent: mu");
        tryLoadingNetwork(JSON_MISSING_PARENT_VERTEX_FILE);
    }

    @Test
    public void loadFailsIfHyperparameterIsMissing() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Failed to create vertex due to missing parent: mu");
        tryLoadingNetwork(JSON_MISSING_HYPERPARAMETER_FILE);
    }

    @Test
    public void loadFailsIfWrongArgumentTypeSpecified() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Incorrect Parameter Type specified.  " +
            "Got: class io.improbable.keanu.tensor.intgr.ScalarIntegerTensor, " +
            "Expected: interface io.improbable.keanu.tensor.dbl.DoubleTensor");
        tryLoadingNetwork(JSON_WRONG_ARGUMENT_TYPE_FILE);
    }

    @Test
    public void loadFailsWithConflictingVertexInfoInValue() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Label and VertexID don't refer to same Vertex: (SomeLabel) " +
            "(id: \"[2]\"\n)");
        tryLoadingNetwork(JSON_INCOMPATIBLE_LABELS_FILE);
    }

    @Test
    public void loadFailsIfValueIsWrongType() throws IOException {
        expectedException.expect(IllegalArgumentException.class);
        expectedException.expectMessage("Non Double Value specified for Double Vertex");
        tryLoadingNetwork(JSON_INCOMPATIBLE_VALUE_FILE);
    }

    private void tryLoadingNetwork(String resourceFileName) throws IOException {
        URL url = Resources.getResource(resourceFileName);
        ByteArrayInputStream input = new ByteArrayInputStream(Resources.toByteArray(url));
        JsonLoader jsonLoader = new JsonLoader();
        BayesianNetwork loadedNetwork = jsonLoader.loadNetwork(input);
    }
}

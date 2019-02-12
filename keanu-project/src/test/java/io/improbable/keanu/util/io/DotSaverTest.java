package io.improbable.keanu.util.io;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.Resources;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

@Slf4j
public class DotSaverTest {

    private static Vertex complexResultVertex;
    private static DotSaver complexNetDotSaver;
    private static ByteArrayOutputStream outputWriter;
    private static final String resourcesFolder = "dotFiles";
    private static final String GAUSSIAN_OUTPUT_FILENAME = resourcesFolder + "/GaussianNodeOutput.dot";
    private static final String TENSOR_OUTPUT_FILENAME = resourcesFolder + "/ConstantTensorIntNodeOutput.dot";
    private static final String SCALAR_OUTPUT_FILENAME = resourcesFolder + "/ConstantScalarIntNodeOutput.dot";
    private static final String LABELLED_OUTPUT_FILENAME = resourcesFolder + "/ConstantLabelledIntNodeOutput.dot";
    private static final String COMPLEX_OUTPUT_FILENAME = resourcesFolder + "/ComplexNetDotOutput.dot";
    private static final String VERTEX_DEGREE1__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree1Output.dot";
    private static final String VERTEX_DEGREE2__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree2Output.dot";
    private static final String OUTPUT_WITH_VALUES_FILENAME = resourcesFolder + "/OutputValuesSetToTrueOutput.dot";
    private static final String OUTPUT_WITH_METADATA_FILENAME = resourcesFolder + "/OutputWithMetadata.dot";
    private static final String OUTPUT_WITH_DISCONNECTED_VERTICES_FILENAME = resourcesFolder + "/OutputWithDisconnectedVertices.dot";
    private static final String VERTEX_DEGREE1__OUTPUT_WITH_DISCONNECTED_VERTICES_FILENAME = resourcesFolder + "/VertexDegree1OutputWithDisconnectedVertices.dot";

    @BeforeClass
    public static void setUpComplexNet() {
        VertexId.ID_GENERATOR.set(0);
        complexResultVertex = ((new GammaVertex(0, 1))
            .lessThan(new ConstantIntegerVertex(-1)))
            .or((new GaussianVertex(0, 1))
            .plus(new ConstantDoubleVertex(5))
            .equalTo(new ConstantDoubleVertex(10)));
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        complexNetDotSaver = new DotSaver(complexNet);
    }

    @Before
    public void resetVertexIdsAndOutputStream() {
        VertexId.ID_GENERATOR.set(0);
        outputWriter = new ByteArrayOutputStream();
    }

    @Test
    public void outputContainsHyperparameters() throws IOException {
        GaussianVertex gaussianV = new GaussianVertex(0, 1);
        BayesianNetwork gaussianNet = new BayesianNetwork(gaussianV.getConnectedGraph());

        DotSaver dotSaver = new DotSaver(gaussianNet);
        dotSaver.save(outputWriter, false);

        String expectedGaussianNodeOutput = readFileToString(GAUSSIAN_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedGaussianNodeOutput);
    }

    @Test
    public void outputtingComplexNet() throws IOException {
        complexNetDotSaver.save(outputWriter, false);
        String expectedComplexOutput = readFileToString(COMPLEX_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedComplexOutput);
    }

    @Test
    public void outputtingVertexDegree1Surroundings() throws IOException {
        complexNetDotSaver.save(outputWriter, Collections.singletonList(complexResultVertex), 1, false);
        String expectedVertexDegree1Output = readFileToString(VERTEX_DEGREE1__OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedVertexDegree1Output);
    }

    @Test
    public void outputtingVertexDegree2Surroundings() throws IOException {
        complexNetDotSaver.save(outputWriter, Collections.singletonList(complexResultVertex), 2, false);
        String expectedVertexDegree2Output = readFileToString(VERTEX_DEGREE2__OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedVertexDegree2Output);
    }

    @Test
    public void dotVertexLabelsAreSetCorrectly() throws IOException {
        int[] intValues = new int[] {1,2,3};
        ConstantIntegerVertex constantIntVertex = new ConstantIntegerVertex(intValues);
        ConstantIntegerVertex constantIntVertex2 = new ConstantIntegerVertex(2);
        IntegerMultiplicationVertex multiplicationVertex = new IntegerMultiplicationVertex(constantIntVertex, constantIntVertex2);
        BayesianNetwork constantIntNet = new BayesianNetwork(multiplicationVertex.getConnectedGraph());

        DotSaver dotSaver = new DotSaver(constantIntNet);
        boolean saveValues = false;

        // Check that class name and annotation (for multiplication vertex) appear in the output.
        dotSaver.save(outputWriter, saveValues);
        String expectedTensorIntNodeOutput = readFileToString(TENSOR_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedTensorIntNodeOutput);

        // Check that vertex label appears in the output if it's set.
        constantIntVertex.setLabel("SomeLabel");
        outputWriter.reset();
        dotSaver.save(outputWriter, saveValues);
        String expectedLabelledIntNodeOutput = readFileToString(LABELLED_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedLabelledIntNodeOutput);

        // Check that value for constant vertices appears in the output if it's set and a scalar.
        constantIntVertex.setValue(42);
        outputWriter.reset();
        dotSaver.save(outputWriter, saveValues);
        String expectedScalarIntNodeOutput = readFileToString(SCALAR_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedScalarIntNodeOutput);
    }

    @Test
    public void valuesAreBeingWrittenOut() throws IOException {
        DoubleVertex unobservedGaussianVertex = new GaussianVertex(0, 1);
        DoubleVertex observedGammaVertex = new GammaVertex(2, 3);
        observedGammaVertex.observe(2.5);
        DoubleVertex gammaMultipliedVertex = observedGammaVertex.times(new ConstantDoubleVertex(4));
        Vertex resultVertex = gammaMultipliedVertex.plus(unobservedGaussianVertex);
        gammaMultipliedVertex.setLabel("Gamma Multiplied");
        DotSaver dotSaver = new DotSaver(new BayesianNetwork(resultVertex.getConnectedGraph()));
        dotSaver.save(outputWriter, true);
        String expectedOutputWithValues = readFileToString(OUTPUT_WITH_VALUES_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedOutputWithValues);
    }

    @Test
    public void metadataIsWrittenOut() throws IOException {
        Map<String, String> metadata = ImmutableMap.of("Author", "Jane Doe", "Version", "V1");
        complexNetDotSaver.save(outputWriter, false, metadata);
        String expectedOutputWithMetadata = readFileToString(OUTPUT_WITH_METADATA_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedOutputWithMetadata);
    }

    @Test
    public void dotSaveShowsDisconnectedVerticesWithDegree1() throws IOException {
        DoubleVertex v1 = new ConstantDoubleVertex(0.);
        DoubleVertex v2 = new ConstantDoubleVertex(1.);
        DoubleVertex gamma1 = new GammaVertex(1., v2);
        gamma1.setLabel("gamma1");
        DoubleVertex gaussian1 = new GaussianVertex(v1, gamma1);
        gaussian1.setLabel("gaussian1");

        DoubleVertex v3 = new ConstantDoubleVertex(0.);
        DoubleVertex v4 = new ConstantDoubleVertex(1.);
        DoubleVertex gamma2 = new GammaVertex(1., v4);
        gamma2.setLabel("gamma2");
        DoubleVertex gaussian2 = new GaussianVertex(v3, gamma2);
        gaussian2.setLabel("gaussian2");

        BayesianNetwork bayesianNetwork = new BayesianNetwork(Arrays.asList(v1, v2, gamma1, gaussian1, v3, v4, gamma2, gaussian2));
        DotSaver dotSaver = new DotSaver(bayesianNetwork);
        dotSaver.save(outputWriter, Arrays.asList(gaussian1, gaussian2), 1, true);
        String expectedOutputWithValues = readFileToString(VERTEX_DEGREE1__OUTPUT_WITH_DISCONNECTED_VERTICES_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedOutputWithValues);
    }

    @Test
    public void dotSaveShowsAllDisconnectedVertices() throws IOException {
        DoubleVertex v1 = new ConstantDoubleVertex(0.);
        DoubleVertex v2 = new ConstantDoubleVertex(1.);
        DoubleVertex gamma1 = new GammaVertex(1., v2);
        gamma1.setLabel("gamma1");
        DoubleVertex gaussian1 = new GaussianVertex(v1, gamma1);
        gaussian1.setLabel("gaussian1");

        DoubleVertex v3 = new ConstantDoubleVertex(0.);
        DoubleVertex v4 = new ConstantDoubleVertex(1.);
        DoubleVertex gamma2 = new GammaVertex(1., v4);
        gamma2.setLabel("gamma2");
        DoubleVertex gaussian2 = new GaussianVertex(v3, gamma2);
        gaussian2.setLabel("gaussian2");

        BayesianNetwork bayesianNetwork = new BayesianNetwork(Arrays.asList(v1, v2, gamma1, gaussian1, v3, v4, gamma2, gaussian2));
        DotSaver dotSaver = new DotSaver(bayesianNetwork);
        dotSaver.save(outputWriter, Arrays.asList(gaussian1, gaussian2), 2, true);
        String expectedOutputWithValues = readFileToString(OUTPUT_WITH_DISCONNECTED_VERTICES_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedOutputWithValues);
    }

    // Need to compare the outputs line by line, as the labels and edges are not written out in a fixed order.
    private void checkDotFilesMatch(String output1, String output2) {

        List<String> output1Lines = Arrays.stream(output1.split("\n"))
            .map(s -> s.replace("\r", ""))
            .collect(Collectors.toList());

        String[] output2Lines = Arrays.stream(output2.split("\n"))
            .map(s -> s.replace("\r", ""))
            .toArray(String[]::new);

        assertThat(output1Lines, containsInAnyOrder(output2Lines));
    }

    private static String readFileToString(String fileOnClassPath) throws IOException {
        URL url = Resources.getResource(fileOnClassPath);
        String fileAsString = Resources.toString(url, Charsets.UTF_8);
        return fileAsString;
    }
}

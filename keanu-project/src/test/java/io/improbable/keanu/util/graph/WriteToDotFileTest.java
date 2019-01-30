package io.improbable.keanu.util.graph;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.util.graph.io.GraphToDot;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastToDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsInAnyOrder;

public class WriteToDotFileTest {

    private static final String resourcesFolder = "dotFiles";
    private static final String GAUSSIAN_OUTPUT_FILENAME = resourcesFolder + "/GaussianNodeOutput.dot";
    private static final String TENSOR_OUTPUT_FILENAME = resourcesFolder + "/ConstantTensorIntNodeOutput.dot";
    private static final String SCALAR_OUTPUT_FILENAME = resourcesFolder + "/ConstantScalarIntNodeOutput.dot";
    private static final String LABELLED_OUTPUT_FILENAME = resourcesFolder + "/ConstantLabelledIntNodeOutput.dot";
    private static final String COMPLEX_OUTPUT_FILENAME = resourcesFolder + "/ComplexNetDotOutput.dot";
    private static final String VERTEX_DEGREE1__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree1Output.dot";
    private static final String VERTEX_DEGREE2__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree2Output.dot";
    private static final String REDUCED_OUTPUT_FILENAME = resourcesFolder + "/ReducedOutput.dot";
    private static final String REDUCED_INTERMEDIATE_OUTPUT_FILENAME = resourcesFolder + "/ReducedIntermediate.dot";
    private static final String SIMPLE_OUTPUT_FILENAME = resourcesFolder + "/SimpleOutput.dot";
    private static final String OUTPUT_WITH_METADATA = resourcesFolder + "/OutputWithMetadata.dot";

    private static Vertex complexResultVertex;
    private static ByteArrayOutputStream outputWriter;

    @BeforeClass
    public static void setUpComplexNet() {
        VertexId.ID_GENERATOR.set(0);
        DoubleVertex n = If.isTrue(new BernoulliVertex(0.5).not()).then(5.0).orElse(1.0);
        IntegerVertex k = If.isTrue(new BernoulliVertex(0.4).not()).then(5).orElse(2);
        DoubleVertex l = new CastToDoubleVertex(k);
        complexResultVertex =
            new GaussianVertex(
                new GammaVertex(0.1, 0.5)
                    .plus(new GaussianVertex(0, n))
                    .plus(new GaussianVertex(1, n))
                    .plus(new GaussianVertex(3, n))
                    .plus(new ConstantDoubleVertex(5)),
                new ConstantDoubleVertex(0.2)
                    .plus(new ConstantDoubleVertex(1.2).times(l)));
        complexResultVertex.observeOwnValue();
    }

    private static String readFileToString(String fileOnClassPath) throws IOException {
        URL url = Resources.getResource(fileOnClassPath);
        String fileAsString = Resources.toString(url, Charsets.UTF_8);
        return fileAsString;
    }

    @Before
    public void resetVertexIdsAndOutputStream() {
        VertexId.ID_GENERATOR.set(0);
        outputWriter = new ByteArrayOutputStream();
    }

    @Test
    public void simpleOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        GraphToDot.write(graph, outputWriter);
        String expectedGaussianNodeOutput = readFileToString(SIMPLE_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedGaussianNodeOutput);
    }

    @Test
    public void valueOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByType().labelVerticesWithValue();
        GraphToDot.write(graph, outputWriter);

        assertThat("Output contains expected string", outputWriter.toString().contains("0 [color=\"#FF0000\"] [label=\"0.5\"]"));
    }

    @Test
    public void reducedOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByState();
        graph.removeDeterministicVertices();
        GraphToDot.write(graph, outputWriter);

        String expectedGaussianNodeOutput = readFileToString(REDUCED_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedGaussianNodeOutput);
    }

    @Test
    public void reducedIntermediateOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByState();
        graph.removeIntermediateVertices();
        GraphToDot.write(graph, outputWriter);

        String expectedGaussianNodeOutput = readFileToString(REDUCED_INTERMEDIATE_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedGaussianNodeOutput);
    }

    @Test
    public void outputContainsHyperparameters() throws IOException {
        GaussianVertex gaussianV = new GaussianVertex(0, 1);
        BayesianNetwork gaussianNet = new BayesianNetwork(gaussianV.getConnectedGraph());

        GraphToDot.write(new VertexGraph(gaussianNet), outputWriter);

        String expectedGaussianNodeOutput = readFileToString(GAUSSIAN_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedGaussianNodeOutput);
    }

    @Test
    public void outputtingComplexNet() throws IOException {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        GraphToDot.write(new VertexGraph(complexNet), outputWriter);
        String expectedComplexOutput = readFileToString(COMPLEX_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedComplexOutput);
    }

    @Test
    public void outputtingVertexDegree1Surroundings() throws IOException {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        GraphToDot.write(new VertexGraph(complexNet, complexResultVertex, 1), outputWriter);
        String expectedVertexDegree1Output = readFileToString(VERTEX_DEGREE1__OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedVertexDegree1Output);
    }

    @Test
    public void outputtingVertexDegree2Surroundings() throws IOException {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        GraphToDot.write(new VertexGraph(complexNet, complexResultVertex, 2), outputWriter);
        String expectedVertexDegree2Output = readFileToString(VERTEX_DEGREE2__OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedVertexDegree2Output);
    }

    @Test
    public void outputtingWithMetadata() throws IOException {
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        VertexGraph g = new VertexGraph(complexNet).putMetadata("author", "Jane Doe").putMetadata("version", "V1").autoPutMetadata();
        g.metadata.remove("timestamp"); // this is non deterministic, so we remove it.
        GraphToDot.write(g, outputWriter);
        String expectedVertexDegree2Output = readFileToString(OUTPUT_WITH_METADATA);
        checkDotFilesMatch(outputWriter.toString(), expectedVertexDegree2Output);
    }

    @Test
    public void dotVertexLabelsAreSetCorrectly() throws IOException {
        int[] intValues = new int[]{1, 2, 3};
        ConstantIntegerVertex constantIntVertex = new ConstantIntegerVertex(intValues);
        ConstantIntegerVertex constantIntVertex2 = new ConstantIntegerVertex(2);
        IntegerMultiplicationVertex multiplicationVertex = new IntegerMultiplicationVertex(constantIntVertex, constantIntVertex2);
        BayesianNetwork constantIntNet = new BayesianNetwork(multiplicationVertex.getConnectedGraph());

        // Check that class name and annotation (for multiplication vertex) appear in the output.
        GraphToDot.write(new VertexGraph(constantIntNet), outputWriter);

        String expectedTensorIntNodeOutput = readFileToString(TENSOR_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedTensorIntNodeOutput);

        // Check that vertex label appears in the output if it's set.
        constantIntVertex.setLabel("SomeLabel");
        outputWriter.reset();
        GraphToDot.write(new VertexGraph(constantIntNet).labelVerticesWithValue(), outputWriter);
        String expectedLabelledIntNodeOutput = readFileToString(LABELLED_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedLabelledIntNodeOutput);

        // Check that value for constant vertices appears in the output if it's set and a scalar.
        constantIntVertex.setValue(42);
        outputWriter.reset();
        GraphToDot.write(new VertexGraph(constantIntNet).labelVerticesWithValue(), outputWriter);

        String expectedScalarIntNodeOutput = readFileToString(SCALAR_OUTPUT_FILENAME);
        checkDotFilesMatch(outputWriter.toString(), expectedScalarIntNodeOutput);
    }

    // Need to compare the outputs line by line, as the labels and edges are not written out in a fixed order.
    private void checkDotFilesMatch(String output1, String output2) {
        List<String> output1Lines = Arrays.stream(output1.split("\n"))
            .map(s -> s.replace("\r", "").trim())
            .collect(Collectors.toList());

        String[] output2Lines = Arrays.stream(output2.split("\n"))
            .map(s -> s.replace("\r", "").trim())
            .toArray(String[]::new);

        assertThat(output1Lines, containsInAnyOrder(output2Lines));
    }
}

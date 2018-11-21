package io.improbable.keanu.util.dot;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertTrue;

public class DotWriterTest {

    private static Vertex complexResultVertex;
    private static DotWriter complexNetDotWriter;
    private static ByteArrayOutputStream outputWriter;
    private static final String resourcesFolder = "./src/test/resources/dotFiles";
    private static final String GAUSSIAN_OUTPUT_FILENAME = resourcesFolder + "/GaussianNodeOutput.dot";
    private static final String TENSOR_OUTPUT_FILENAME = resourcesFolder + "/ConstantTensorIntNodeOutput.dot";
    private static final String SCALAR_OUTPUT_FILENAME = resourcesFolder + "/ConstantScalarIntNodeOutput.dot";
    private static final String LABELLED_OUTPUT_FILENAME = resourcesFolder + "/ConstantLabelledIntNodeOutput.dot";
    private static final String COMPLEX_OUTPUT_FILENAME = resourcesFolder + "/ComplexNetDotOutput.dot";
    private static final String VERTEX_DEGREE1__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree1Output.dot";
    private static final String VERTEX_DEGREE2__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree2Output.dot";
    private static final String OUTPUT_WITH_VALUES_FILENAME = resourcesFolder + "/OutputValuesSetToTrueOutput.dot";


    @BeforeClass
    public static void setUpComplexNet() {
        complexResultVertex = (new ConstantBoolVertex(true))
            .or((new GaussianVertex(0, 1))
            .plus(new ConstantDoubleVertex(5))
            .equalTo(new ConstantDoubleVertex(10)));
        BayesianNetwork complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
        complexNetDotWriter = new DotWriter(complexNet);
    }

    @Before
    public void resetVertexIdGenerator() {
        VertexId.ID_GENERATOR.set(0);
        outputWriter = new ByteArrayOutputStream();
    }

    @Test
    public void outputContainsHyperparameters() throws IOException {
        GaussianVertex gaussianV = new GaussianVertex(0, 1);
        BayesianNetwork gaussianNet = new BayesianNetwork(gaussianV.getConnectedGraph());

        DotWriter dotWriter = new DotWriter(gaussianNet);
        dotWriter.save(outputWriter, false);

        String expectedGaussianNodeOutput = new String(Files.readAllBytes(Paths.get(GAUSSIAN_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedGaussianNodeOutput));
    }

    @Test
    public void outputtingComplexNet() throws IOException {
        complexNetDotWriter.save(outputWriter, false);
        String expectedComplexOutput = new String(Files.readAllBytes(Paths.get(COMPLEX_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedComplexOutput));
    }

    @Test
    public void outputtingVertexDegree1Surroundings() throws IOException {
        complexNetDotWriter.save(outputWriter, complexResultVertex, 1, false);
        String expectedVertexDegree1Output = new String(Files.readAllBytes(Paths.get(VERTEX_DEGREE1__OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedVertexDegree1Output));
    }

    @Test
    public void outputtingVertexDegree2Surroundings() throws IOException {
        complexNetDotWriter.save(outputWriter, complexResultVertex, 2, false);
        String expectedVertexDegree2Output = new String(Files.readAllBytes(Paths.get(VERTEX_DEGREE2__OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedVertexDegree2Output));
    }

    @Test
    public void dotVertexLabelsAreSetCorrectly() throws IOException {
        int[] intValues = new int[] {1,2,3};
        ConstantIntegerVertex constantIntVertex = new ConstantIntegerVertex(intValues);
        ConstantIntegerVertex constantIntVertex2 = new ConstantIntegerVertex(2);
        IntegerMultiplicationVertex multiplicationVertex = new IntegerMultiplicationVertex(constantIntVertex, constantIntVertex2);
        BayesianNetwork constantIntNet = new BayesianNetwork(multiplicationVertex.getConnectedGraph());

        DotWriter dotWriter = new DotWriter(constantIntNet);
        boolean saveValues = false;

        // Check that class name and annotation (for multiplication vertex) appear in the output.
        dotWriter.save(outputWriter, saveValues);
        String expectedTensorIntNodeOutput = new String(Files.readAllBytes(Paths.get(TENSOR_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedTensorIntNodeOutput));

        // Check that vertex label appears in the output if it's set.
        constantIntVertex.setLabel("SomeLabel");
        outputWriter.reset();
        dotWriter.save(outputWriter, saveValues);
        String expectedLabelledIntNodeOutput = new String(Files.readAllBytes(Paths.get(LABELLED_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedLabelledIntNodeOutput));

        // Check that value for constant vertices appears in the output if it's set and a scalar.
        constantIntVertex.setValue(42);
        outputWriter.reset();
        dotWriter.save(outputWriter, saveValues);
        String expectedScalarIntNodeOutput = new String(Files.readAllBytes(Paths.get(SCALAR_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedScalarIntNodeOutput));
    }

    @Test
    public void valuesAreBeingWrittenOut() throws IOException {
        DoubleVertex unobservedGaussianVertex = new GaussianVertex(0, 1);
        DoubleVertex observedGammaVertex = new GammaVertex(2, 3);
        observedGammaVertex.observe(2.5);
        DoubleVertex gammaMultipliedVertex = observedGammaVertex.times(new ConstantDoubleVertex(4));
        Vertex resultVertex = gammaMultipliedVertex.plus(unobservedGaussianVertex);
        gammaMultipliedVertex.setLabel("Gamma Multiplied");
        DotWriter dotWriter = new DotWriter(new BayesianNetwork(resultVertex.getConnectedGraph()));
        dotWriter.save(outputWriter, true);
        String expectedOutputWithValues = new String(Files.readAllBytes(Paths.get(OUTPUT_WITH_VALUES_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedOutputWithValues));
    }

    // Need to compare the outputs line by line, as the labels and edges are not written out in a fixed order.
    private boolean dotFilesAreEqual(String output1, String output2) {
        List<String> output1Lines = Arrays.asList(output1.split("\n"));
        List<String> output2Lines = Arrays.asList(output2.split("\n"));

        if (output1Lines.size() != output2Lines.size()) {
            return false;
        }

        // Don't care about line ending format.
        output1Lines = output1Lines.stream().map(line -> line.replace("\r", "")).collect(Collectors.toList());
        output2Lines = output2Lines.stream().map(line -> line.replace("\r", "")).collect(Collectors.toList());

        for (String line : output1Lines) {
            if (!output2Lines.contains(line)) {
                return false;
            }
        }
        return true;
    }
}

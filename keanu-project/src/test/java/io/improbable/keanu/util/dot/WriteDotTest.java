package io.improbable.keanu.util.dot;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import org.apache.commons.io.FileUtils;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayWithSize;

public class WriteDotTest {

    private static BayesianNetwork complexNet;
    private static Vertex complexResultVertex;
    private static final String resourcesFolder = "./src/test/resources/dotFiles";
    private static final String GAUSSIAN_OUTPUT_FILENAME = resourcesFolder + "/GaussianNodeOutput.dot";
    private static final String TENSOR_OUTPUT_FILENAME = resourcesFolder + "/ConstantTensorIntNodeOutput.dot";
    private static final String SCALAR_OUTPUT_FILENAME = resourcesFolder + "/ConstantScalarIntNodeOutput.dot";
    private static final String LABELLED_OUTPUT_FILENAME = resourcesFolder + "/ConstantLabelledIntNodeOutput.dot";
    private static final String COMPLEX_OUTPUT_FILENAME = resourcesFolder + "/ComplexNetDotOutput.dot";
    private static final String VERTEX_DEGREE1__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree1Output.dot";
    private static final String VERTEX_DEGREE2__OUTPUT_FILENAME = resourcesFolder + "/VertexDegree2Output.dot";


    @BeforeClass
    public static void setUpComplexNet() {
        complexResultVertex = (new ConstantBoolVertex(true)).or((new GaussianVertex(0, 1)).plus(new ConstantDoubleVertex(5)).equalTo(new ConstantDoubleVertex(10)));
        complexNet = new BayesianNetwork(complexResultVertex.getConnectedGraph());
    }

    @Before
    public void resetVertexIdGenerator() {
        VertexId.ID_GENERATOR.set(0);
    }

    @Test
    public void filesGetWrittenAndAreNotOverwritten() throws IOException {
        String fileName = "dottestfolder/test_file.dot";
        File outputFile = new File(fileName);
        FileUtils.deleteDirectory(outputFile.getParentFile());

        WriteDot.outputDot(fileName, complexNet);
        assertTrue(outputFile.exists());

        WriteDot.outputDot(fileName, complexNet);
        assertThat(outputFile.getParentFile().listFiles(), arrayWithSize(2));

        FileUtils.deleteDirectory(outputFile.getParentFile());
    }

    @Test
    public void outputContainsHyperparameters() throws IOException {
        GaussianVertex gaussianV = new GaussianVertex(0, 1);
        BayesianNetwork gaussianNet = new BayesianNetwork(gaussianV.getConnectedGraph());

        StringWriter outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, gaussianNet);

        String expectedGaussianNodeOutput = new String(Files.readAllBytes(Paths.get(GAUSSIAN_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedGaussianNodeOutput));
    }

    @Test
    public void outputtingComplexNet() throws IOException {
        StringWriter outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, complexNet);
        String expectedComplexOutput = new String(Files.readAllBytes(Paths.get(COMPLEX_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedComplexOutput));
    }

    @Test
    public void outputtingVertexDegree1Surroundings() throws IOException {
        StringWriter outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, complexResultVertex, 1);
        String expectedVertexDegree1Output = new String(Files.readAllBytes(Paths.get(VERTEX_DEGREE1__OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedVertexDegree1Output));
    }

    @Test
    public void outputtingVertexDegree2Surroundings() throws IOException {
        StringWriter outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, complexResultVertex, 2);
        String expectedVertexDegree2Output = new String(Files.readAllBytes(Paths.get(VERTEX_DEGREE2__OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedVertexDegree2Output));
    }

    @Test
    public void dotVertexLabelsAreSetCorrectly() throws IOException {
        StringWriter outputWriter = new StringWriter();

        int[] intValues = new int[] {1,2,3};
        ConstantIntegerVertex constantIntVertex = new ConstantIntegerVertex(intValues);
        ConstantIntegerVertex constantIntVertex2 = new ConstantIntegerVertex(2);
        IntegerMultiplicationVertex multiplicationVertex = new IntegerMultiplicationVertex(constantIntVertex, constantIntVertex2);
        BayesianNetwork constantIntNet = new BayesianNetwork(multiplicationVertex.getConnectedGraph());

        // Check that class name and annotation (for multiplication vertex) appear in the output.
        WriteDot.outputDot(outputWriter, constantIntNet);
        String expectedTensorIntNodeOutput = new String(Files.readAllBytes(Paths.get(TENSOR_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedTensorIntNodeOutput));

        // Check that vertex label appears in the output if it's set.
        constantIntVertex.setLabel("SomeLabel");
        outputWriter.getBuffer().setLength(0);
        WriteDot.outputDot(outputWriter, constantIntNet);
        String expectedLabelledIntNodeOutput = new String(Files.readAllBytes(Paths.get(LABELLED_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedLabelledIntNodeOutput));

        // Check that value for constant vertices appears in the output if it's set and a scalar.
        constantIntVertex.setValue(42);
        outputWriter.getBuffer().setLength(0);
        WriteDot.outputDot(outputWriter, constantIntNet);
        String expectedScalarIntNodeOutput = new String(Files.readAllBytes(Paths.get(SCALAR_OUTPUT_FILENAME)), Charset.defaultCharset());
        assertTrue(dotFilesAreEqual(outputWriter.toString(), expectedScalarIntNodeOutput));
    }

    // Need to compare the outputs line by line, as the labels and edges are not written out in a fixed order.
    private boolean dotFilesAreEqual(String output1, String output2) {
        String[] output1Lines = output1.split("\n");
        String[] output2Lines = output2.split("\n");

        if (output1Lines.length != output2Lines.length) {
            return false;
        }

        for (String line : output1Lines) {
            if (!Arrays.asList(output2Lines).contains(line)) {
                return false;
            }
        }
        return true;
    }
}

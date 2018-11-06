package io.improbable.keanu.util.dot;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.apache.commons.io.FileUtils;
import org.junit.*;

import java.io.File;
import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.arrayWithSize;

public class WriteDotTest {

    private static BayesianNetwork gaussianNet;
    private static BayesianNetwork booleanNet;
    private static final String testFileDir = "dottestdir";
    private static String gaussianDotOutput;
    private static String boolDotOutput;
    private static String vertexDegree1DotOutput;
    private static String vertexDegree2DotOutput;
    private static final String boolVertexLabel = "/BoolResult";
    private static GaussianVertex gaussianV;
    private static DoubleVertex constantDoubleV;

    @BeforeClass
    public static void setup() {

        double mu = 0;
        double sigma = 1;
        gaussianV = new GaussianVertex(mu, sigma);
        constantDoubleV = new ConstantDoubleVertex(5.43);
        DoubleVertex guassianGraphResult = gaussianV.multiply(constantDoubleV);
        gaussianNet = new BayesianNetwork(guassianGraphResult.getConnectedGraph());

        Writer outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, gaussianNet);
        gaussianDotOutput = outputWriter.toString();

        BoolVertex boolGraphResult = new ConstantBoolVertex(true).or(new ConstantIntegerVertex(5).greaterThan(new ConstantDoubleVertex(5.43)));
        boolGraphResult.setLabel(boolVertexLabel);
        booleanNet = new BayesianNetwork(boolGraphResult.getConnectedGraph());
        outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, booleanNet);
        boolDotOutput = outputWriter.toString();

        outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, gaussianV, 1);
        vertexDegree1DotOutput = outputWriter.toString();

        outputWriter = new StringWriter();
        WriteDot.outputDot(outputWriter, gaussianV, 2);
        vertexDegree2DotOutput = outputWriter.toString();
    }

    @Test
    public void filesGetWrittenAndAreNotOverwritten() throws IOException {
        String fileName = "dottestfolder/test_file.dot";
        File outputFile = new File(fileName);
        FileUtils.deleteDirectory(outputFile.getParentFile());

        WriteDot.outputDot(fileName, gaussianNet);
        assertTrue(outputFile.exists());

        WriteDot.outputDot(fileName, booleanNet);
        assertThat(outputFile.getParentFile().listFiles(), arrayWithSize(2));

        FileUtils.deleteDirectory(outputFile.getParentFile());
    }

    @Test
    public void outputContainsAllVertices() {
        List<Vertex> allGaussianVertices = gaussianNet.getAllVertices();

        for (Vertex v : allGaussianVertices) {
            assertTrue(gaussianDotOutput.contains(v.getId().hashCode() + ""));
        }

        List<Vertex> allBoolVertices = booleanNet.getAllVertices();

        for (Vertex v : allBoolVertices) {
            assertTrue(boolDotOutput.contains(v.getId().hashCode() + ""));
        }
    }

    @Test
    public void outputContainsAllEdges() {
        List<Vertex> allGaussianVertices = gaussianNet.getAllVertices();

        for (Vertex v : allGaussianVertices) {
            for (Object vChild : v.getChildren()) {
                assertTrue(gaussianDotOutput.contains("<" + v.hashCode() + "> -> <" + vChild.hashCode() + ">"));
            }
        }

        List<Vertex> allBoolVertices = booleanNet.getAllVertices();

        for (Vertex v : allBoolVertices) {
            for (Object vChild : v.getChildren()) {
                assertTrue(boolDotOutput.contains("<" + v.hashCode() + "> -> <" + ((Vertex) vChild).hashCode() + ">"));
            }
        }
    }

    @Test
    public void outputContainsHyperparameters() throws IOException {
        assertTrue(gaussianDotOutput.contains("[label=mu]"));
        assertTrue(gaussianDotOutput.contains("[label=sigma]"));
    }

    @Test
    public void outputContainsVertexLabels() {
        assertTrue(boolDotOutput.contains(boolVertexLabel));
    }

    @Test
    public void outputContainsAnnotations() {
        String multiplyVertexAnnotation = MultiplicationVertex.class.getAnnotation(DisplayInformationForOutput.class).displayName();
        assertTrue(gaussianDotOutput.contains(multiplyVertexAnnotation));
    }

    @Test
    public void outputContainsClassName() {
        assertTrue(gaussianDotOutput.contains(GaussianVertex.class.getSimpleName()));
    }

    @Test
    public void outputVertexWithDegree1() {
        assertTrue(vertexDegree1DotOutput.contains(gaussianV.getId().hashCode() + ""));
        assertFalse(vertexDegree1DotOutput.contains(constantDoubleV.getId().hashCode() + ""));
        // Vertices within 1 degree form the specified vertex cover less of a graph than vertices within two degrees.
        assertTrue(org.apache.commons.lang3.StringUtils.countMatches(vertexDegree1DotOutput, "->") < org.apache.commons.lang3.StringUtils.countMatches(vertexDegree2DotOutput, "->"));
    }

    @Test
    public void outputVertexWithDegree2() {
        // For this net vertices within 2 connections from the Gaussian vertex cover the entire net.
        assertTrue(vertexDegree2DotOutput.equals(gaussianDotOutput));
    }

    @AfterClass
    public static void cleanup() throws IOException {
        File testDir = new File(testFileDir);
        FileUtils.deleteDirectory(testDir);
    }


}

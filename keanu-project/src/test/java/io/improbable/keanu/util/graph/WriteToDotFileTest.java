package io.improbable.keanu.util.graph;

import io.improbable.keanu.util.graph.io.GraphToDot;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GammaVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class WriteToDotFileTest {

    private static Vertex complexResultVertex;
    private static ByteArrayOutputStream outputWriter;

    @BeforeClass
    public static void setUpComplexNet() {
        VertexId.ID_GENERATOR.set(0);
        complexResultVertex =
            new GaussianVertex(
                new GammaVertex(0.1, 0.5)
                    .plus(new GaussianVertex(0, 1))
                    .plus(new GaussianVertex(0, 1))
                    .plus(new GaussianVertex(0, 1))
                    .plus(new ConstantDoubleVertex(5)),
                new ConstantDoubleVertex(0.2)
                    .plus(new ConstantDoubleVertex(1.2)));
        complexResultVertex.observeOwnValue();
    }

    @Before
    public void resetVertexIdsAndOutputStream() {
        VertexId.ID_GENERATOR.set(0);
        outputWriter = new ByteArrayOutputStream();
    }

    @Test
    public void simpleOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByType();
        GraphToDot.write(graph, outputWriter);
        System.out.println(outputWriter.toString());
    }

    @Test
    public void valueOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByType().labelEdgesWithParameters();
        GraphToDot.write(graph, outputWriter);
        System.out.println(outputWriter.toString());
    }

    @Test
    public void reducedOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByState();
        graph.removeDeterministicVertices();
        graph.labelVerticesWithValue();
        GraphToDot.write(graph, outputWriter);
        System.out.println(outputWriter.toString());
    }

    @Test
    public void reducedIntermediateOutput() throws IOException {
        VertexGraph graph = new VertexGraph(complexResultVertex);
        graph.labelEdgesWithParameters().colorVerticesByState();
        graph.removeIntermediateVertices();
        GraphToDot.write(graph, outputWriter);
        System.out.println(outputWriter.toString());
    }
}

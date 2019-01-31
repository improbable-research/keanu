package io.improbable.keanu.util.graph;

import io.improbable.keanu.util.graph.io.GraphToDot;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;


public class NamespacesTest {

    private static Vertex outA;
    private static ByteArrayOutputStream outputWriter;

    @BeforeClass
    public static void setUpComplexNet() {
        VertexId.ID_GENERATOR.set(0);
        DoubleVertex initial1 = new ConstantDoubleVertex(1.0);
        DoubleVertex initial5 = new ConstantDoubleVertex(5.0);
        DoubleVertex initial10 = new ConstantDoubleVertex(10.0);

        initial1.setLabel(new VertexLabel("1", "initial"));
        initial5.setLabel(new VertexLabel("5", "initial"));
        initial10.setLabel(new VertexLabel("10", "initial"));

        DoubleVertex intermediateA = initial1.plus(initial5);
        intermediateA.setLabel(new VertexLabel("A", "intermediate"));
        DoubleVertex intermediateB = initial1.div(initial5);
        intermediateB.setLabel(new VertexLabel("B", "intermediate"));
        DoubleVertex intermediateC = initial10.plus(initial5);
        intermediateC.setLabel(new VertexLabel("C", "intermediate"));
        DoubleVertex intermediateD = initial5.times(initial10);
        intermediateD.setLabel(new VertexLabel("D", "intermediate"));

        outA = new GaussianVertex(intermediateA, intermediateC);
        outA.setLabel(new VertexLabel("A", "out"));
    }

    @Before
    public void resetVertexIdsAndOutputStream() {
        VertexId.ID_GENERATOR.set(0);
        outputWriter = new ByteArrayOutputStream();
    }

    @Test
    public void simpleOutput() throws IOException {
        VertexGraph graph = new VertexGraph(outA).labelConstantVerticesWithValue();
        GraphToDot.write(graph, outputWriter);
        int lines = outputWriter.toString().split("\n").length;
        assertEquals(20, lines);
    }

    @Test
    public void colorByNamespace() {
        VertexGraph graph = new VertexGraph(outA).colorVerticesByNamespace();
        Set<String> colors = graph.getNodes().stream().map((n) -> n.details.get("color")).collect(Collectors.toSet());
        GraphToDot.write( graph , System.out );
        String s = String.join(",", colors);
        assertEquals("There are three distinct colours " + s, 3, colors.size());
    }

    @Test
    public void removeIntermediates() throws IOException {
        VertexGraph graph = new VertexGraph(outA).removeNamespace("intermediate").colorVerticesByNamespace();
        Set<String> colors = graph.getNodes().stream().map((n) -> n.details.get("color")).collect(Collectors.toSet());
        GraphToDot.write( graph , System.out );
        String s = String.join(",", colors);
        assertEquals("There are two distinct colours " + s, 2, colors.size());
    }
}

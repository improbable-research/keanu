package io.improbable.keanu.backend.keanu;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

public class KeanuComputableGraphTest {

    private DoubleVertex A;
    private DoubleVertex B;

    private static final String A_LABEL = "A";
    private static final String B_LABEL = "B";

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0).setLabel(A_LABEL);
        B = new GaussianVertex(0.0, 1.0).setLabel(B_LABEL);
    }

    @Test
    public void canConvertSimpleGraphToComputationalGraph() {
        String cLabel = "C";
        DoubleVertex C = A.plus(B);
        C.getValue();
        C.setLabel(cLabel);

        List<Vertex> toposortedGraph = C.getConnectedGraph().stream()
            .sorted(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()))
            .collect(Collectors.toList());

        KeanuComputableGraph computableGraph = new KeanuComputableGraph(toposortedGraph, ImmutableSet.of(C));

        computableGraph.compute(ImmutableMap.of(
            A.getReference(), DoubleTensor.scalar(1),
            B.getReference(), DoubleTensor.scalar(1)
        ));

        DoubleTensor evaluatedC = (DoubleTensor) computableGraph.compute(ImmutableMap.of(
            A.getReference(), DoubleTensor.scalar(2),
            B.getReference(), DoubleTensor.scalar(3)
        )).get(C.getReference());

        assertEquals(DoubleTensor.scalar(5), evaluatedC);
    }

    @Test
    public void canConvertGraphToComputationalGraph() {
        DoubleVertex C = A.plus(B);
        DoubleVertex D = C.plus(A);
        DoubleVertex E = C.plus(B);
        DoubleVertex F = D.plus(E);
        F.getValue();

        List<Vertex> toposortedGraph = F.getConnectedGraph().stream()
            .sorted(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()))
            .collect(Collectors.toList());

        KeanuComputableGraph computableGraph = new KeanuComputableGraph(toposortedGraph, ImmutableSet.of(F));

        computableGraph.compute(ImmutableMap.of(
            A.getReference(), DoubleTensor.scalar(1),
            B.getReference(), DoubleTensor.scalar(1)
        ));

        DoubleTensor evaluatedC = (DoubleTensor) computableGraph.compute(ImmutableMap.of(
            A.getReference(), DoubleTensor.scalar(2),
            B.getReference(), DoubleTensor.scalar(3)
        )).get(F.getReference());

        assertEquals(DoubleTensor.scalar(15), evaluatedC);
    }

}

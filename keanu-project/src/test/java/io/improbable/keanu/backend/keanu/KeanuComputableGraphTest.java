package io.improbable.keanu.backend.keanu;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class KeanuComputableGraphTest {

    GaussianVertex A;
    GaussianVertex B;

    @Before
    public void setup() {
        A = new GaussianVertex(0.0, 1.0);
        A.setLabel("A");
        B = new GaussianVertex(0.0, 1.0);
        B.setLabel("B");
    }

    @Test
    public void canConvertSimpleGraphToComputationalGraph() {
        DoubleVertex C = A.plus(B);
        C.setLabel("C");

        KeanuComputableGraph computableGraph = KeanuGraphConverter.convert(C.getConnectedGraph());

        DoubleTensor evaluatedC = computableGraph.compute(ImmutableMap.of(
            "A", DoubleTensor.scalar(2),
            "B", DoubleTensor.scalar(3)
        ), "C");

        assertEquals(DoubleTensor.scalar(5), evaluatedC);
    }
}

package io.improbable.keanu.backend.keanu;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class KeanuComputableGraphTest {

    private GaussianVertex A;
    private GaussianVertex B;

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
        C.setLabel(cLabel);

        KeanuComputableGraph computableGraph = KeanuGraphConverter.convert(C.getConnectedGraph());

        DoubleTensor evaluatedC = computableGraph.compute(ImmutableMap.of(
            A.getReference(), DoubleTensor.scalar(2),
            B.getReference(), DoubleTensor.scalar(3)
        ), C.getReference());

        assertEquals(DoubleTensor.scalar(5), evaluatedC);
    }
}

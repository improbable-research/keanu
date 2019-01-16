package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Rule;
import org.junit.Test;

import static com.google.common.collect.ImmutableMap.of;
import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void canMaintainStateInGraph() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        DoubleTensor result = graph.compute(of(
            A.getReference(), A.getValue(),
            B.getReference(), B.getValue()
        ), C.getReference());

        assertEquals(C.eval(), result);

        DoubleTensor nextBValue = DoubleTensor.create(new double[]{0.1, 0.1, 0.1, 0.1}, 2, 2);
        DoubleTensor resultAfterRun = graph.compute(of(
            B.getReference(), nextBValue
        ), C.getReference());

        B.setValue(nextBValue);

        assertEquals(C.eval(), resultAfterRun);
    }

}

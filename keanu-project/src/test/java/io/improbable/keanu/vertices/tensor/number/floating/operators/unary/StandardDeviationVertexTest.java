package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers.finiteDifferenceMatchesElementwise;

public class StandardDeviationVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        UnaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::standardDeviation, FloatingPointTensorVertex::standardDeviation);
    }

    @Test
    @Ignore
    public void changesMatchGradient() {
        finiteDifferenceMatchesElementwise(DoubleVertex::standardDeviation);
    }
}

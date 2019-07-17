package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import org.junit.Test;

public class StandardizeVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        UnaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::standardize, FloatingPointTensorVertex::standardize);
    }
}

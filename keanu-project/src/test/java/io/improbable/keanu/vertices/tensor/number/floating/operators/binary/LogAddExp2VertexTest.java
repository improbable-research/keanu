package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import org.junit.Ignore;
import org.junit.Test;

public class LogAddExp2VertexTest {

    @Test
    public void doesOperateOnMatrix() {
        BinaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::logAddExp2, FloatingPointTensorVertex::logAddExp2);
    }

    @Test
    @Ignore
    public void finiteDifferenceMatchesElementwise() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesElementwise(DoubleVertex::logAddExp2);
    }

    @Test
    @Ignore
    public void finiteDifferenceMatchesSimpleBroadcast() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesBroadcast(DoubleVertex::logAddExp2);
    }
}

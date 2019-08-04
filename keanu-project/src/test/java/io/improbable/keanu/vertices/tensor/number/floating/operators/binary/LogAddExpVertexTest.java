package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Ignore;
import org.junit.Test;

public class LogAddExpVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        BinaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::logAddExp, FloatingPointTensorVertex::logAddExp);
    }

    @Test
    @Ignore
    public void finiteDifferenceMatchesElementwise() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesElementwise(DoubleVertex::logAddExp);
    }

    @Test
    @Ignore
    public void finiteDifferenceMatchesSimpleBroadcast() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesBroadcast(DoubleVertex::logAddExp);
    }

}

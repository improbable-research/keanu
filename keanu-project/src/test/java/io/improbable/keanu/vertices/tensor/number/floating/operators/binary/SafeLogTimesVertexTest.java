package io.improbable.keanu.vertices.tensor.number.floating.operators.binary;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.BinaryOperationTestHelpers;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsWithRespectTo;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class SafeLogTimesVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        BinaryOperationTestHelpers.operatesOnInput(FloatingPointTensor::safeLogTimes, FloatingPointTensorVertex::safeLogTimes);
    }

    @Test
    public void finiteDifferenceMatchesElementwise() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesElementwise(DoubleVertex::safeLogTimes, 0.1, 2.0, -1.0, 1.0);
    }

    @Test
    public void finiteDifferenceMatchesSimpleBroadcast() {
        BinaryOperationTestHelpers.finiteDifferenceMatchesBroadcast(DoubleVertex::safeLogTimes, 0.1, 2.0, -1.0, 1.0);
    }

    @Test
    public void finiteDifferenceMatchesWhenYIsZero() {
        UniformVertex X = new UniformVertex(0.1, 2.0);
        UniformVertex Y = new UniformVertex(-1.0, 1.0);
        Y.setValue(DoubleTensor.scalar(0));

        DoubleVertex output = X.safeLogTimes(Y);

        PartialsWithRespectTo dOutputWrtY = Differentiator.forwardModeAutoDiff(Y, output);
        assertThat(dOutputWrtY.of(output), valuesAndShapesMatch(DoubleTensor.scalar(Double.NaN)));

        PartialsOf dOutputWrtXY = Differentiator.reverseModeAutoDiff(output, X, Y);
        assertThat(dOutputWrtXY.withRespectTo(Y), valuesAndShapesMatch(DoubleTensor.scalar(Double.NaN)));

    }
}
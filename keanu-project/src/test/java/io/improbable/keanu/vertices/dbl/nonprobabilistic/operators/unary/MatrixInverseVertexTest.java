package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static org.junit.Assert.assertEquals;

public class MatrixInverseVertexTest {

    private static final int NUM_ITERATIONS = 10;

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonSquareInput() {
        DoubleTensor matrix = DoubleTensor.arange(1, 9).reshape(4, 2);

        shouldReject(matrix);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonMatrix() {
        DoubleTensor tensor = DoubleTensor.arange(1,9).reshape(2, 2, 2);

        shouldReject(tensor);
    }

    private void shouldReject(DoubleTensor tensor) {
        DoubleVertex inputVertex = new ConstantDoubleVertex(tensor);
        DoubleVertex invertVertex = inputVertex.matrixInverse();
    }

    @Test
    public void canTakeInverseCorrectly() {
        DoubleTensor matrix = DoubleTensor.arange(1,5).reshape(2, 2);
        DoubleVertex inputVertex = new ConstantDoubleVertex(matrix);
        DoubleVertex inverseVertex = inputVertex.matrixInverse();

        inverseVertex.lazyEval();

        DoubleTensor expected = DoubleTensor.create(new double[]{-2, 1, 1.5, -0.5}, 2, 2);

        assertEquals(expected, inverseVertex.getValue());
    }

    @Test
    public void canCalculateDiffCorrectly() {
        DoubleVertex matrix = new UniformVertex(1.0, 100.0);
        matrix.setValue(DoubleTensor.arange(1, 5).reshape(2, 2));
        MatrixInverseVertex inverse = matrix.matrixInverse();

        inverse.lazyEval();

        DoubleTensor inverseWrtMatrix = inverse.getDerivativeWrtLatents().withRespectTo(matrix);
        DoubleTensor reverseInverseWrtMatrix = Differentiator.reverseModeAutoDiff(inverse, matrix).withRespectTo(matrix);

        DoubleTensor expectedInverseWrtMatrix = DoubleTensor.create(new double[]{
            -4.0, 3.0,
            2.0, -1.5,
            2.0, -1.0,
            -1.0, 0.5,
            3.0, -2.25,
            -1.0, 0.75,
            -1.5, 0.75,
            0.5, -0.25},
            new long[]{2, 2, 2, 2}
        );

        assertEquals(expectedInverseWrtMatrix, inverseWrtMatrix);
        assertEquals(expectedInverseWrtMatrix, reverseInverseWrtMatrix);
    }

    @Test
    public void inverseMultipliedEqualsIdentity() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{4, 4}, -20.0, 20.0);
        DoubleVertex inverseVertex = inputVertex.matrixInverse();
        MatrixMultiplicationVertex multiplied = inverseVertex.matrixMultiply(inputVertex);

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            inputVertex.setValue(inputVertex.sample());
            DoubleTensor result = multiplied.eval();

            assertEquals(result, DoubleTensor.eye(4));

            DoubleTensor changeInMultipliedWrtInput =
                multiplied.getDerivativeWrtLatents().withRespectTo(inputVertex);
            DoubleTensor reverseOutputWrtInput =
                Differentiator.reverseModeAutoDiff(multiplied, inputVertex).withRespectTo(inputVertex);
            assertEquals(changeInMultipliedWrtInput.pow(2.0).sum(), 0.0, 1e-10);
            assertEquals(reverseOutputWrtInput.pow(2.0).sum(), 0.0, 1e-10);
        }
    }

    @Test
    public void scalarTensorsInvertCorrectly() {
        DoubleTensor oneByOneMatrix = new ScalarDoubleTensor(2.0);
        DoubleVertex input = new ConstantDoubleVertex(oneByOneMatrix);
        DoubleVertex inverse = input.matrixInverse();

        inverse.lazyEval();

        assertEquals(0.5, inverse.getValue().scalar(), 1e-6);
    }

    @Test
    public void inverseDifferenceMatchesGradient() {
        DoubleVertex inputVertex = new UniformVertex(new long[]{3, 3}, 1.0, 25.0);
        MatrixInverseVertex invertVertex = inputVertex.matrixInverse();

        finiteDifferenceMatchesForwardAndReverseModeGradient(
            ImmutableList.of(inputVertex), invertVertex, 0.001, 1e-5);
    }

}

package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class MatrixInverseVertexTest {

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonSquareInput() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);

        shouldReject(matrix);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonMatrix() {
        DoubleTensor tensor = DoubleTensor.create(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);

        shouldReject(tensor);
    }

    private void shouldReject(DoubleTensor tensor) {
        DoubleVertex inputVertex = new ConstantDoubleVertex(tensor);
        DoubleVertex invertVertex = new MatrixInverseVertex(inputVertex);

        invertVertex.lazyEval();
    }

    @Test
    public void canTakeInverseCorrectly() {
        DoubleTensor matrix = DoubleTensor.create(new double[]{3, 4, 5, 6}, 2, 2);
        DoubleVertex inputVertex = new ConstantDoubleVertex(matrix);
        DoubleVertex inverseVertex = new MatrixInverseVertex(inputVertex);

        inverseVertex.lazyEval();

        DoubleTensor expected = DoubleTensor.create(new double[]{-3, 2, 2.5, -1.5}, 2, 2);

        assertEquals(expected, inverseVertex.getValue());
    }

    @Test
    public void inverseMultipliedEqualsIdentity() {
        DoubleVertex inputVertex = new UniformVertex(new int[]{3,3}, 0.0, 20.0);
        DoubleVertex inverseVertex = new MatrixInverseVertex(inputVertex);
        DoubleVertex multiplied = inverseVertex.matrixMultiply(inputVertex);

        DoubleTensor result = multiplied.eval();

        assertEquals(result, DoubleTensor.eye(3));

        DoubleTensor changeInMultipliedWrtInput =
            multiplied.getDualNumber().getPartialDerivatives().withRespectTo(inputVertex);
        System.out.println(changeInMultipliedWrtInput);
        assertEquals(changeInMultipliedWrtInput.sum(), 0.0, 1e-10);
    }

    @Test
    public void canCalculateDualCorrectly() {
        DoubleVertex matrix = new UniformVertex(1.0, 100.0);
        matrix.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));
        DoubleVertex inverse = new MatrixInverseVertex(matrix);

        DoubleTensor partialD = inverse.getDualNumber().getPartialDerivatives().withRespectTo(matrix);

        System.out.println(partialD);
    }
}

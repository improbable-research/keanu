package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.TensorTestOperations.finiteDifferenceMatchesGradient;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwo2x2MatrixVertexValues;
import static io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.BinaryOperationTestHelpers.operatesOnTwoScalarVertexValues;

public class MaxVertexTest {

    @Test
    public void maxOfTwoScalarValues() {
        operatesOnTwoScalarVertexValues(2.0, 3.0, 3.0, DoubleVertex::max);
    }

    @Test
    public void maxOfTwoMatrixVertexValues() {
        operatesOnTwo2x2MatrixVertexValues(
            new double[]{1.0, 2.0, 6.0, 4.0},
            new double[]{2.0, 4.0, 3.0, 4.0},
            new double[]{2.0, 4.0, 6.0, 4.0},
            DoubleVertex::max
        );
    }

    @Test
    public void changesMatchGradient() {
        DoubleVertex A = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        A.setValue(DoubleTensor.create(1, 2, 3, 4, 5, 6, 7, 8));
        DoubleVertex B = new UniformVertex(new int[]{2, 2, 2}, -10.0, 10.0);
        B.setValue(DoubleTensor.create(8, 7, 6, 5, 4, 3, 2, 1));
        DoubleVertex C = DoubleVertex.max(A, B);

        finiteDifferenceMatchesGradient(ImmutableList.of(A, B), C, 1e-6, 1e-10, true);
    }

}
